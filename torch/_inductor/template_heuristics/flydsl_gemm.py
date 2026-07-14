from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FlyDSLHGemmConfig:
    TILE_M: int
    TILE_N: int = 128
    TILE_K: int = 64
    STAGES: int = 2
    SPLIT_K: int = 1
    BLOCK_M_WARPS: int = 1
    BLOCK_N_WARPS: int = 4
    BLOCK_K_WARPS: int = 1
    B_TO_LDS: bool = True


def get_default_hgemm_configs() -> list[FlyDSLHGemmConfig]:
    return [
        FlyDSLHGemmConfig(TILE_M=32),
        FlyDSLHGemmConfig(TILE_M=64),
        FlyDSLHGemmConfig(TILE_M=128),
    ]


def get_hgemm_configs(m: int, n: int, k: int) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    for config in get_default_hgemm_configs():
        if config.TILE_M > max(128, m):
            continue
        if k % config.SPLIT_K != 0:
            continue
        if (k // config.SPLIT_K) // config.TILE_K < config.STAGES:
            continue
        configs.append(
            {
                **asdict(config),
            }
        )
    return configs


def get_grouped_hgemm_configs(m: int, n: int, k: int) -> list[dict[str, object]]:
    """Return grouped GEMM configs.

    The grouped path receives B as [G, K, N].  Keep the default search focused
    on direct strided-B loads: the grouped gather-to-LDS path is correct, but
    is slower on the small/irregular grouped workloads this template targets.
    """
    candidates = [
        # Small-M grouped/decode configs.  These reduce wasted work when each
        # group has far fewer than 32 rows.
        FlyDSLHGemmConfig(
            TILE_M=16, TILE_N=64, BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, B_TO_LDS=False
        ),
        FlyDSLHGemmConfig(
            TILE_M=16, TILE_N=128, BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, B_TO_LDS=False
        ),
        FlyDSLHGemmConfig(
            TILE_M=32, TILE_N=64, BLOCK_M_WARPS=1, BLOCK_N_WARPS=2, B_TO_LDS=False
        ),
        FlyDSLHGemmConfig(
            TILE_M=32, TILE_N=256, BLOCK_M_WARPS=1, BLOCK_N_WARPS=4, B_TO_LDS=False
        ),
        # Baseline hgemm-like configs.  The grouped path currently reuses the
        # hgemm store/LDS layout, which is validated for TILE_N=128.
        FlyDSLHGemmConfig(TILE_M=32, TILE_N=128, B_TO_LDS=False),
        FlyDSLHGemmConfig(TILE_M=64, TILE_N=128, B_TO_LDS=False),
        FlyDSLHGemmConfig(TILE_M=64, TILE_N=256, B_TO_LDS=False),
        FlyDSLHGemmConfig(TILE_M=128, TILE_N=128, B_TO_LDS=False),
    ]
    configs: list[dict[str, object]] = []
    for config in candidates:
        if config.TILE_M > max(128, m):
            continue
        if n < config.TILE_N or n % config.TILE_N != 0:
            continue
        if k % config.SPLIT_K != 0:
            continue
        if (k // config.SPLIT_K) // config.TILE_K < config.STAGES:
            continue
        configs.append({**asdict(config)})
    return configs
