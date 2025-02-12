from pathlib import Path

# Local imports
from rfb_utils import SimulationType
from rfb_campaign import run_sim
from make_inputs import subdir_tbl

def run_cnvg_flow_steady(G: int, nzs: list[int]):
    """Run all the simulations for the flow convergence study to steady state"""
    # Find all the simulation directories
    dir_campaign: Path = Path('01_cnvg_flow_steady')
    sub_dir: str = subdir_tbl[G]
    dir_geom: Path = dir_campaign / sub_dir

    # Assemble list of all candidate simulation directories
    dir_sims_all: list[Path] = list(dir_geom.glob('n???'))
    dir_sims_all.sort()
    
    # List simulations before running them
    print('Found the following simulation directories available:')
    for dir_sim in dir_sims_all:
        print(dir_sim)

    # Filter down to only the requested simulations
    print('Processing the following simulation directories:')
    dir_sims: list[Path] = list()
    for nz in nzs:
        dir_sim = dir_geom / f'n{nz:03d}'
        if dir_sim in dir_sims_all:
            dir_sims.append(dir_sim)
            print(dir_sim)

    # Run the simulations in the specified order
    sim_type: SimulationType = SimulationType.Flow
    dim: int = 3
    for dir_sim in dir_sims:
        run_sim(dir_sim=dir_sim, sim_type=sim_type, dim=dim)

def run_cnvg_flow_fixed(G: int, nzs: list[int]):
    """Run all the simulations for the flow convergence study to a fixed time"""
    # Find all the simulation directories
    dir_campaign: Path = Path('02_cnvg_flow_tfix')
    sub_dir: str = subdir_tbl[G]
    dir_geom: Path = dir_campaign / sub_dir

    # Assemble list of all candidate simulation directories
    dir_sims_all: list[Path] = list(dir_geom.glob('n???'))
    dir_sims_all.sort()
    
    # List simulations before running them
    print('Found the following simulation directories available:')
    for dir_sim in dir_sims_all:
        print(dir_sim)

    # Filter down to only the requested simulations
    print('Processing the following simulation directories:')
    dir_sims: list[Path] = list()
    for nz in nzs:
        dir_sim = dir_geom / f'n{nz:03d}'
        if dir_sim in dir_sims_all:
            dir_sims.append(dir_sim)
            print(dir_sim)

    # Run the simulations in the specified order
    sim_type: SimulationType = SimulationType.Flow
    dim: int = 3
    for dir_sim in dir_sims:
        run_sim(dir_sim=dir_sim, sim_type=sim_type, dim=dim)

def run_cnvg_flow_mesh(G: int, nzs: list[int]):
    """Run all the simulations for the flow convergence study to a fixed time"""
    # Find all the simulation directories
    dir_campaign: Path = Path('04_cnvg_flow_mesh')
    sub_dir: str = subdir_tbl[G]
    dir_geom: Path = dir_campaign / sub_dir

    # Assemble list of all candidate simulation directories
    dir_sims_all: list[Path] = list(dir_geom.glob('n???'))
    dir_sims_all.sort()
    
    # List simulations before running them
    print('Found the following simulation directories available:')
    for dir_sim in dir_sims_all:
        print(dir_sim)

    # Filter down to only the requested simulations
    print('Processing the following simulation directories:')
    dir_sims: list[Path] = list()
    for nz in nzs:
        dir_sim = dir_geom / f'n{nz:03d}'
        if dir_sim in dir_sims_all:
            dir_sims.append(dir_sim)
            print(dir_sim)

    # Run the simulations in the specified order
    sim_type: SimulationType = SimulationType.Flow
    dim: int = 3
    for dir_sim in dir_sims:
        run_sim(dir_sim=dir_sim, sim_type=sim_type, dim=dim)

def main():
    G: int = 2
    nzs: list[int] = [8, 16, 32, 64]
    run_cnvg_flow_mesh(G=G, nzs=nzs)

if __name__ == '__main__':
    main()
