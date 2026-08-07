#pragma once

/// A from-scratch implementation of the PDX vector similarity search algorithm:
///   L. Kuffo, E. Krippner, P. Boncz. "PDX: A Data Layout for Vector Similarity search".
///   SIGMOD 2025. https://arxiv.org/abs/2503.04422
///
/// Implemented here:
/// - The PDX layout: vectors are stored in blocks of 64, dimension-major within a block,
///   so that one dimension of all vectors in a block is contiguous in memory.
/// - PDXearch: distances are accumulated dimension-by-dimension for a whole block in
///   tight loops over independent accumulators (auto-vectorizable, no horizontal
///   reduction), with pruning of vectors whose partial distance already exceeds the
///   current k-th best distance. Dimensions are fetched in exponentially growing steps
///   (2, 4, 8, ...) and the pruning bound is re-evaluated after each step.
/// - PDX-BOND dimension ordering: dimension zones are ranked per query by the
///   distance-to-means criterion, so that the most discriminative dimensions for this
///   query are scanned first. No preprocessing of the data is required.
///
/// The implementation is tailored to ClickHouse vector similarity granules: search is
/// single-threaded and there is no IVF clustering (a granule holds a bounded number of
/// vectors, so the whole granule is treated as one collection of blocks).
///
/// The pruning is exact for L2: the partial sum of squared differences never decreases,
/// so it is a valid lower bound of the final distance. Cosine distance is reduced to L2
/// on the unit sphere: vectors are normalized at build time, the query is normalized at
/// search time, and cosineDistance = ||q - v||^2 / 2 for unit q, v.
///
/// This header deliberately does not throw exceptions. Precondition violations are
/// programming errors and are guarded by assertions; recoverable errors (corrupt
/// serialized data) are reported to the caller, which is expected to throw a proper
/// ClickHouse exception.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace PDX
{

/// Guards against absurd index definitions; the integration layer validates user input
/// against this limit before constructing an index.
inline constexpr size_t PDX_MAX_DIMS = 16384;

/// "Blocks of 64 vectors perform best across all architectures as the SIMD registers
/// holding the resulting distances are recycled through the entire block processing
/// without intermediate LOAD/STORE instructions." (PDX paper, Section 6.2)
inline constexpr size_t BLOCK_SIZE = 64;

/// Granularity of PDX-BOND dimension zones. Zones are consecutive runs of dimensions
/// which are ranked per query by the distance-to-means criterion; keeping dimensions
/// within a zone sequential preserves sequential memory access inside a block.
inline constexpr size_t ZONE_SIZE = 16;

/// First step of the WARMUP phase; subsequent steps double: 2, 4, 8, ... (Section 4)
inline constexpr size_t FIRST_DIMENSION_STEP = 2;

/// Once at most this many vectors of a block survive, switch from processing all lanes
/// to processing only the surviving positions (the PRUNE phase of PDXearch).
inline constexpr size_t SPARSE_SWITCH_THRESHOLD = BLOCK_SIZE / 4;

enum class DistanceMetric : uint8_t
{
    L2SQ = 0,
    COSINE = 1,
};

struct KNNCandidate
{
    uint32_t index;
    float distance;
};

/// Counters exposed to the integration layer for observability of the pruning.
struct SearchStats
{
    /// Number of (dimension, vector) value visits performed by the distance kernels.
    /// A full scan without pruning visits num_vectors * num_dimensions values.
    uint64_t scanned_values = 0;
    /// Number of vectors discarded by the pruning bound before all their dimensions
    /// were visited.
    uint64_t pruned_vectors = 0;
};

struct PDXIndexConfig
{
    uint32_t num_dimensions = 0;
    DistanceMetric distance_metric = DistanceMetric::L2SQ;
};

class PDXIndexF32
{
public:
    explicit PDXIndexF32(PDXIndexConfig config_)
        : config(config_)
    {
        assert(config.num_dimensions > 0);
        assert(config.num_dimensions <= PDX_MAX_DIMS);
    }

    /// Transposes row-major input vectors into the PDX layout and computes the
    /// per-dimension means used by the PDX-BOND dimension ordering.
    /// For the cosine metric the vectors are normalized first (see header comment).
    void buildIndex(const float * embeddings, size_t num_embeddings_)
    {
        assert(num_embeddings_ == 0 || embeddings != nullptr);
        assert(num_embeddings_ <= std::numeric_limits<uint32_t>::max());

        num_embeddings = num_embeddings_;
        const size_t dims = config.num_dimensions;
        const size_t num_blocks = numBlocks();

        /// Padded lanes of the last block stay zero; they are never reported because
        /// bookkeeping below only covers real lanes.
        block_data.assign(num_blocks * dims * BLOCK_SIZE, 0.0f);
        means.assign(dims, 0.0f);
        block_means.assign(num_blocks * dims, 0.0f);

        if (num_embeddings == 0)
            return;

        std::vector<double> mean_accumulators(dims, 0.0);
        std::vector<double> block_mean_accumulators(num_blocks * dims, 0.0);

        for (size_t row = 0; row < num_embeddings; ++row)
        {
            const float * vector = embeddings + row * dims;

            float norm_inv = 1.0f;
            if (config.distance_metric == DistanceMetric::COSINE)
            {
                float sq_norm = 0.0f;
                for (size_t dim = 0; dim < dims; ++dim)
                    sq_norm += vector[dim] * vector[dim];
                const float norm = std::sqrt(sq_norm);
                if (norm > 0.0f)
                    norm_inv = 1.0f / norm;
            }

            const size_t block = row / BLOCK_SIZE;
            const size_t lane = row % BLOCK_SIZE;
            float * block_base = block_data.data() + block * dims * BLOCK_SIZE;

            for (size_t dim = 0; dim < dims; ++dim)
            {
                const float value = vector[dim] * norm_inv;
                block_base[dim * BLOCK_SIZE + lane] = value;
                mean_accumulators[dim] += static_cast<double>(value);
                block_mean_accumulators[block * dims + dim] += static_cast<double>(value);
            }
        }

        for (size_t dim = 0; dim < dims; ++dim)
            means[dim] = static_cast<float>(mean_accumulators[dim] / static_cast<double>(num_embeddings));

        for (size_t block = 0; block < num_blocks; ++block)
        {
            const double count = static_cast<double>(blockCount(block));
            for (size_t dim = 0; dim < dims; ++dim)
                block_means[block * dims + dim] = static_cast<float>(block_mean_accumulators[block * dims + dim] / count);
        }
    }

    std::vector<KNNCandidate> search(const float * query, size_t knn, SearchStats * stats = nullptr) const
    {
        assert(query != nullptr);

        if (knn == 0 || num_embeddings == 0)
            return {};

        SearchContext context = prepareSearch(query);

        /// Visit blocks in ascending order of the distance from the query to the block
        /// centroid, so that the k-th best bound tightens as early as possible and the
        /// remaining blocks prune aggressively. This adapts the paper's IVF strategy of
        /// processing the most promising clusters first to the flat block collection of
        /// a granule.
        for (size_t block : blockVisitOrder(context))
            searchBlock(block, blockCount(block), /*initial_positions=*/nullptr, knn, context);

        if (stats)
            *stats = context.stats;
        return finishSearch(context, knn);
    }

    /// Same search pipeline restricted to the given row ids. All blocks start directly
    /// in the PRUNE (positions) phase with the passing rows of that block.
    std::vector<KNNCandidate> filteredSearch(
        const float * query,
        size_t knn,
        const std::vector<size_t> & passing_row_ids,
        SearchStats * stats = nullptr) const
    {
        assert(query != nullptr);

        if (knn == 0 || num_embeddings == 0 || passing_row_ids.empty())
            return {};

        SearchContext context = prepareSearch(query);

        /// Group passing rows by block. passing_row_ids is not required to be sorted.
        std::vector<std::vector<uint32_t>> positions_per_block(numBlocks());
        for (size_t row : passing_row_ids)
        {
            if (row >= num_embeddings)
                continue;
            positions_per_block[row / BLOCK_SIZE].push_back(static_cast<uint32_t>(row % BLOCK_SIZE));
        }

        for (size_t block : blockVisitOrder(context))
        {
            if (positions_per_block[block].empty())
                continue;
            std::sort(positions_per_block[block].begin(), positions_per_block[block].end());
            searchBlock(block, blockCount(block), &positions_per_block[block], knn, context);
        }

        if (stats)
            *stats = context.stats;
        return finishSearch(context, knn);
    }

    /// Serializes the index. The writer callback must write exactly `size` bytes.
    void serialize(const std::function<void(const char *, size_t)> & write) const
    {
        writeValue(write, SERIALIZATION_MAGIC);
        writeValue(write, SERIALIZATION_VERSION);
        writeValue(write, static_cast<uint8_t>(config.distance_metric));
        writeValue(write, config.num_dimensions);
        writeValue(write, static_cast<uint64_t>(num_embeddings));
        write(reinterpret_cast<const char *>(means.data()), means.size() * sizeof(float));
        write(reinterpret_cast<const char *>(block_means.data()), block_means.size() * sizeof(float));
        write(reinterpret_cast<const char *>(block_data.data()), block_data.size() * sizeof(float));
    }

    /// Deserializes an index previously written by serialize. The reader callback must
    /// read exactly `size` bytes. Returns false and sets `error` on malformed input;
    /// the caller is expected to throw. The index must have been constructed with the
    /// same dimensions and metric that were used for serialization.
    bool deserialize(const std::function<void(char *, size_t)> & read, std::string & error)
    {
        uint32_t magic = 0;
        uint32_t version = 0;
        uint8_t metric_raw = 0;
        uint32_t dims = 0;
        uint64_t count = 0;

        readValue(read, magic);
        if (magic != SERIALIZATION_MAGIC)
        {
            error = "invalid PDX index magic number";
            return false;
        }

        readValue(read, version);
        if (version != SERIALIZATION_VERSION)
        {
            error = "unsupported PDX index version " + std::to_string(version);
            return false;
        }

        readValue(read, metric_raw);
        if (metric_raw != static_cast<uint8_t>(config.distance_metric))
        {
            error = "PDX index metric mismatch (expected " + std::to_string(static_cast<uint8_t>(config.distance_metric))
                + ", got " + std::to_string(metric_raw) + ")";
            return false;
        }

        readValue(read, dims);
        if (dims != config.num_dimensions)
        {
            error = "PDX index dimensions mismatch (expected " + std::to_string(config.num_dimensions)
                + ", got " + std::to_string(dims) + ")";
            return false;
        }

        readValue(read, count);
        if (count > std::numeric_limits<uint32_t>::max())
        {
            error = "invalid number of vectors in PDX index: " + std::to_string(count);
            return false;
        }

        num_embeddings = count;
        means.resize(config.num_dimensions);
        block_means.resize(numBlocks() * config.num_dimensions);
        block_data.resize(numBlocks() * config.num_dimensions * BLOCK_SIZE);
        read(reinterpret_cast<char *>(means.data()), means.size() * sizeof(float));
        read(reinterpret_cast<char *>(block_means.data()), block_means.size() * sizeof(float));
        read(reinterpret_cast<char *>(block_data.data()), block_data.size() * sizeof(float));
        return true;
    }

    uint32_t getNumDimensions() const
    {
        return config.num_dimensions;
    }

    size_t getNumVectors() const
    {
        return num_embeddings;
    }

    size_t getInMemorySizeInBytes() const
    {
        return sizeof(*this)
            + block_data.size() * sizeof(float)
            + means.size() * sizeof(float)
            + block_means.size() * sizeof(float);
    }

private:
    static constexpr uint32_t SERIALIZATION_MAGIC = 0x32584450; /// "PDX2"
    static constexpr uint32_t SERIALIZATION_VERSION = 2;

    struct SearchContext
    {
        /// Query, normalized for the cosine metric.
        std::vector<float> query;
        /// Dimension visit order: zones ranked by the distance-to-means criterion.
        std::vector<uint32_t> dim_order;
        /// Top-k candidates as a binary max-heap over (distance, index).
        std::vector<KNNCandidate> heap;
        SearchStats stats;
    };

    size_t numBlocks() const
    {
        return (num_embeddings + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    /// Number of real (non-padding) lanes in a block.
    size_t blockCount(size_t block) const
    {
        const size_t begin = block * BLOCK_SIZE;
        return std::min(BLOCK_SIZE, num_embeddings - begin);
    }

    SearchContext prepareSearch(const float * query) const
    {
        const size_t dims = config.num_dimensions;

        SearchContext context;
        context.query.assign(query, query + dims);

        if (config.distance_metric == DistanceMetric::COSINE)
        {
            float sq_norm = 0.0f;
            for (float value : context.query)
                sq_norm += value * value;
            const float norm = std::sqrt(sq_norm);
            if (norm > 0.0f)
            {
                const float norm_inv = 1.0f / norm;
                for (float & value : context.query)
                    value *= norm_inv;
            }
        }

        /// PDX-BOND: rank dimension zones by how far the query is from the per-dimension
        /// means; the farther the query value, the faster that dimension accumulates
        /// distance and the earlier vectors can be pruned. (Section 5 of the paper)
        const size_t num_zones = (dims + ZONE_SIZE - 1) / ZONE_SIZE;
        std::vector<float> zone_scores(num_zones, 0.0f);
        for (size_t dim = 0; dim < dims; ++dim)
            zone_scores[dim / ZONE_SIZE] += std::fabs(context.query[dim] - means[dim]);

        std::vector<uint32_t> zone_order(num_zones);
        std::iota(zone_order.begin(), zone_order.end(), 0);
        std::sort(zone_order.begin(), zone_order.end(), [&](uint32_t lhs, uint32_t rhs)
        {
            if (zone_scores[lhs] == zone_scores[rhs])
                return lhs < rhs;
            return zone_scores[lhs] > zone_scores[rhs];
        });

        context.dim_order.reserve(dims);
        for (uint32_t zone : zone_order)
        {
            const size_t zone_begin = static_cast<size_t>(zone) * ZONE_SIZE;
            const size_t zone_end = std::min(zone_begin + ZONE_SIZE, dims);
            for (size_t dim = zone_begin; dim < zone_end; ++dim)
                context.dim_order.push_back(static_cast<uint32_t>(dim));
        }

        return context;
    }

    /// Blocks sorted by ascending distance from the query to the block centroid.
    std::vector<size_t> blockVisitOrder(const SearchContext & context) const
    {
        const size_t dims = config.num_dimensions;
        const size_t num_blocks = numBlocks();

        std::vector<float> block_scores(num_blocks, 0.0f);
        for (size_t block = 0; block < num_blocks; ++block)
        {
            const float * centroid = block_means.data() + block * dims;
            float score = 0.0f;
            for (size_t dim = 0; dim < dims; ++dim)
            {
                const float diff = context.query[dim] - centroid[dim];
                score += diff * diff;
            }
            block_scores[block] = score;
        }

        std::vector<size_t> order(num_blocks);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs)
        {
            if (block_scores[lhs] == block_scores[rhs])
                return lhs < rhs;
            return block_scores[lhs] < block_scores[rhs];
        });
        return order;
    }

    /// The pruning bound: the distance of the current k-th best candidate, or infinity
    /// while fewer than k candidates have been collected (the START phase: the first
    /// block is effectively a linear scan because nothing can be pruned yet).
    static float currentThreshold(const SearchContext & context, size_t knn)
    {
        if (context.heap.size() < knn)
            return std::numeric_limits<float>::infinity();
        return context.heap.front().distance;
    }

    static bool heapOrder(const KNNCandidate & lhs, const KNNCandidate & rhs)
    {
        if (lhs.distance == rhs.distance)
            return lhs.index < rhs.index;
        return lhs.distance < rhs.distance;
    }

    static void tryAddCandidate(SearchContext & context, size_t knn, uint32_t row, float distance)
    {
        auto worse = [](const KNNCandidate & lhs, const KNNCandidate & rhs) { return heapOrder(lhs, rhs); };

        KNNCandidate candidate{row, distance};
        if (context.heap.size() < knn)
        {
            context.heap.push_back(candidate);
            std::push_heap(context.heap.begin(), context.heap.end(), worse);
            return;
        }

        if (heapOrder(candidate, context.heap.front()))
        {
            std::pop_heap(context.heap.begin(), context.heap.end(), worse);
            context.heap.back() = candidate;
            std::push_heap(context.heap.begin(), context.heap.end(), worse);
        }
    }

    /// PDXearch over one block. `initial_positions == nullptr` starts in the dense
    /// (WARMUP) phase over all lanes; otherwise the block starts directly in the
    /// sparse (PRUNE) phase restricted to the given lanes.
    void searchBlock(
        size_t block,
        size_t block_count,
        const std::vector<uint32_t> * initial_positions,
        size_t knn,
        SearchContext & context) const
    {
        const size_t dims = config.num_dimensions;
        const float * block_base = block_data.data() + block * dims * BLOCK_SIZE;
        const uint32_t row_base = static_cast<uint32_t>(block * BLOCK_SIZE);

        alignas(64) float accumulators[BLOCK_SIZE] = {};

        /// Survivor positions live on the stack; a block never has more than
        /// BLOCK_SIZE lanes, so no allocation is needed on this hot path.
        uint32_t positions[BLOCK_SIZE];
        size_t positions_count = 0;
        bool sparse = false;
        if (initial_positions)
        {
            for (uint32_t lane : *initial_positions)
                positions[positions_count++] = lane;
            sparse = true;
        }

        const size_t initial_count = sparse ? positions_count : block_count;

        size_t processed_dims = 0;
        size_t step = FIRST_DIMENSION_STEP;

        while (processed_dims < dims)
        {
            const size_t chunk_end = std::min(processed_dims + step, dims);

            if (!sparse)
            {
                /// WARMUP: process all lanes of the block. The inner loop runs over
                /// contiguous memory with BLOCK_SIZE independent accumulators and no
                /// horizontal reduction, which makes it auto-vectorizable. Pruned lanes
                /// are recomputed too - that waste is what the positions phase below
                /// eliminates once few enough lanes survive.
                for (size_t i = processed_dims; i < chunk_end; ++i)
                {
                    const size_t dim = context.dim_order[i];
                    const float query_value = context.query[dim];
                    const float * values = block_base + dim * BLOCK_SIZE;
                    for (size_t lane = 0; lane < BLOCK_SIZE; ++lane)
                    {
                        const float diff = query_value - values[lane];
                        accumulators[lane] += diff * diff;
                    }
                }
                context.stats.scanned_values += (chunk_end - processed_dims) * block_count;
            }
            else
            {
                /// PRUNE: "we break off distance computations by skipping the already
                /// discarded vectors" - only surviving positions are visited.
                for (size_t i = processed_dims; i < chunk_end; ++i)
                {
                    const size_t dim = context.dim_order[i];
                    const float query_value = context.query[dim];
                    const float * values = block_base + dim * BLOCK_SIZE;
                    for (size_t position = 0; position < positions_count; ++position)
                    {
                        const uint32_t lane = positions[position];
                        const float diff = query_value - values[lane];
                        accumulators[lane] += diff * diff;
                    }
                }
                context.stats.scanned_values += (chunk_end - processed_dims) * positions_count;
            }

            processed_dims = chunk_end;
            step *= 2;

            if (processed_dims >= dims)
                break;

            /// Re-evaluate the pruning bound after each step. The partial L2 sum is a
            /// monotone lower bound of the final distance, so anything above the
            /// current k-th best distance can never enter the result (Section 2.3).
            const float threshold = currentThreshold(context, knn);
            if (std::isinf(threshold))
                continue;

            if (!sparse)
            {
                size_t survivors = 0;
                for (size_t lane = 0; lane < block_count; ++lane)
                    if (accumulators[lane] <= threshold)
                        ++survivors;

                if (survivors == 0)
                    break;

                if (survivors <= SPARSE_SWITCH_THRESHOLD)
                {
                    /// "maintaining the count of the remaining vectors and their
                    /// positions within the block. Then, the following distance
                    /// calculations are performed using this positions array." (Section 4)
                    for (size_t lane = 0; lane < block_count; ++lane)
                        if (accumulators[lane] <= threshold)
                            positions[positions_count++] = static_cast<uint32_t>(lane);
                    sparse = true;
                }
            }
            else
            {
                size_t write_index = 0;
                for (size_t position = 0; position < positions_count; ++position)
                {
                    const uint32_t lane = positions[position];
                    if (accumulators[lane] <= threshold)
                        positions[write_index++] = lane;
                }
                positions_count = write_index;

                if (positions_count == 0)
                    break;
            }
        }

        /// Report survivors that were scanned to the end; their partial sums are now
        /// complete distances.
        size_t survivors = 0;
        if (processed_dims >= dims)
        {
            const float threshold = currentThreshold(context, knn);
            if (sparse)
            {
                for (size_t position = 0; position < positions_count; ++position)
                {
                    const uint32_t lane = positions[position];
                    if (accumulators[lane] <= threshold || context.heap.size() < knn)
                        tryAddCandidate(context, knn, row_base + lane, accumulators[lane]);
                }
                survivors = positions_count;
            }
            else
            {
                for (size_t lane = 0; lane < block_count; ++lane)
                {
                    if (accumulators[lane] <= threshold || context.heap.size() < knn)
                        tryAddCandidate(context, knn, row_base + static_cast<uint32_t>(lane), accumulators[lane]);
                }
                survivors = block_count;
            }
        }

        context.stats.pruned_vectors += initial_count - std::min(survivors, initial_count);
    }

    std::vector<KNNCandidate> finishSearch(SearchContext & context, size_t knn) const
    {
        std::vector<KNNCandidate> result = std::move(context.heap);
        std::sort(result.begin(), result.end(), heapOrder);
        if (result.size() > knn)
            result.resize(knn);

        /// Cosine distance was searched as L2 on the unit sphere:
        /// ||q - v||^2 = 2 - 2 * (q . v), hence cosineDistance = 1 - (q . v) = L2^2 / 2.
        if (config.distance_metric == DistanceMetric::COSINE)
        {
            for (KNNCandidate & candidate : result)
                candidate.distance /= 2.0f;
        }

        return result;
    }

    template <typename T>
    static void writeValue(const std::function<void(const char *, size_t)> & write, const T & value)
    {
        write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    template <typename T>
    static void readValue(const std::function<void(char *, size_t)> & read, T & value)
    {
        read(reinterpret_cast<char *>(&value), sizeof(T));
    }

    PDXIndexConfig config;
    size_t num_embeddings = 0;
    /// PDX layout: value of dimension `d` of vector `row` lives at
    /// block_data[(block * num_dimensions + d) * BLOCK_SIZE + lane], where
    /// block = row / BLOCK_SIZE and lane = row % BLOCK_SIZE. Padding lanes are zero.
    std::vector<float> block_data;
    /// Per-dimension means of the stored (possibly normalized) values, for PDX-BOND.
    std::vector<float> means;
    /// Per-block per-dimension means (block centroids), for the block visit order.
    std::vector<float> block_means;
};

}
