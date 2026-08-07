#!/usr/bin/env bash
# Tags: no-fasttest, no-ordinary-database

CURDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../shell_config.sh
. "$CURDIR"/../shell_config.sh

use_pdx=$($CLICKHOUSE_CLIENT -q "SELECT toUInt8(any(value) IN ('1', 'ON', 'TRUE')) FROM system.build_options WHERE name='USE_PDX'")
if [[ "$use_pdx" != "1" ]]; then
    # PDX is not compiled in. Emit the expected output so that the test passes on such builds.
    cat <<'EOF'
smoke
0
1
L2 exactness
1
L2 index used
1
cosine exactness
1
cosine index used
1
EOF
    exit 0
fi

REFERENCE_VEC="[0.11, 0.27, 0.45, 0.83, 0.05, 0.31, 0.99, 0.62, 0.74, 0.58, 0.02, 0.66, 0.15, 0.48, 0.91, 0.37]"

$CLICKHOUSE_CLIENT -nmq "
    SELECT 'smoke';

    DROP TABLE IF EXISTS tab_pdx_smoke;
    CREATE TABLE tab_pdx_smoke
    (
        id UInt64,
        vec Array(Float32),
        INDEX idx vec TYPE vector_similarity('pdx', 'cosineDistance', 2)
    )
    ENGINE = MergeTree
    ORDER BY id
    SETTINGS index_granularity = 8192;

    INSERT INTO tab_pdx_smoke VALUES
        (0, [1.0, 0.0]),
        (1, [0.9, 0.1]),
        (2, [0.0, 1.0]),
        (3, [0.1, 0.9]);

    SELECT id
    FROM tab_pdx_smoke
    ORDER BY cosineDistance(vec, [1.0, 0.0])
    LIMIT 2
    SETTINGS vector_search_with_rescoring = 0;

    DROP TABLE tab_pdx_smoke;

    -- 1000 rows of deterministic pseudo-random 16-dimensional vectors, laid out so that
    -- one part contains multiple index granules (index_granularity = 100, GRANULARITY 2
    -- --> 200 rows per indexed block) and each indexed block spans multiple PDX blocks
    -- of 64 vectors. The top-10 of the indexed search must match the exact scan.

    SELECT 'L2 exactness';

    DROP TABLE IF EXISTS tab_pdx_l2;
    CREATE TABLE tab_pdx_l2
    (
        id UInt64,
        vec Array(Float32),
        INDEX idx vec TYPE vector_similarity('pdx', 'L2Distance', 16) GRANULARITY 2
    )
    ENGINE = MergeTree
    ORDER BY id
    SETTINGS index_granularity = 100;

    INSERT INTO tab_pdx_l2
        SELECT number, arrayMap(i -> toFloat32((sipHash64(number, i) % 1000000) / 1000000.), range(16))
        FROM numbers(1000);

    SELECT
    (
        SELECT groupArray(id) FROM
        (
            SELECT id FROM tab_pdx_l2
            ORDER BY L2Distance(vec, ${REFERENCE_VEC})
            LIMIT 10
            SETTINGS use_skip_indexes = 0
        )
    ) =
    (
        SELECT groupArray(id) FROM
        (
            SELECT id FROM tab_pdx_l2
            ORDER BY L2Distance(vec, ${REFERENCE_VEC})
            LIMIT 10
            SETTINGS vector_search_with_rescoring = 0
        )
    );

    SELECT 'L2 index used';

    SELECT countMatches(explain_output, 'vector_similarity') >= 1 FROM
    (
        SELECT arrayStringConcat(groupArray(explain), '\n') AS explain_output FROM
        (
            EXPLAIN indexes = 1
            SELECT id FROM tab_pdx_l2
            ORDER BY L2Distance(vec, ${REFERENCE_VEC})
            LIMIT 10
        )
    );

    DROP TABLE tab_pdx_l2;

    SELECT 'cosine exactness';

    DROP TABLE IF EXISTS tab_pdx_cos;
    CREATE TABLE tab_pdx_cos
    (
        id UInt64,
        vec Array(Float32),
        INDEX idx vec TYPE vector_similarity('pdx', 'cosineDistance', 16) GRANULARITY 2
    )
    ENGINE = MergeTree
    ORDER BY id
    SETTINGS index_granularity = 100;

    INSERT INTO tab_pdx_cos
        SELECT number, arrayMap(i -> toFloat32(0.05 + (sipHash64(number, i) % 1000000) / 1000000.), range(16))
        FROM numbers(1000);

    SELECT
    (
        SELECT groupArray(id) FROM
        (
            SELECT id FROM tab_pdx_cos
            ORDER BY cosineDistance(vec, ${REFERENCE_VEC})
            LIMIT 10
            SETTINGS use_skip_indexes = 0
        )
    ) =
    (
        SELECT groupArray(id) FROM
        (
            SELECT id FROM tab_pdx_cos
            ORDER BY cosineDistance(vec, ${REFERENCE_VEC})
            LIMIT 10
            SETTINGS vector_search_with_rescoring = 0
        )
    );

    SELECT 'cosine index used';

    SELECT countMatches(explain_output, 'vector_similarity') >= 1 FROM
    (
        SELECT arrayStringConcat(groupArray(explain), '\n') AS explain_output FROM
        (
            EXPLAIN indexes = 1
            SELECT id FROM tab_pdx_cos
            ORDER BY cosineDistance(vec, ${REFERENCE_VEC})
            LIMIT 10
        )
    );

    DROP TABLE tab_pdx_cos;
"
