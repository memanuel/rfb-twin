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

def run_cnvg_react(G: int, nzs: list[int]):
    """Run all the simulations for the reaction convergence study"""
    # Find all the simulation directories
    # dir_campaign: Path = Path('03_cnvg_react')
    dir_campaign: Path = Path('05_cnvg_react_mesh')
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
    sim_type: SimulationType = SimulationType.React
    dim: int = 3
    for dir_sim in dir_sims:
        run_sim(dir_sim=dir_sim, sim_type=sim_type, dim=dim)
        
def main():
    # Geometry to run
    G: int = 3
    # Factors of 30 starting with 5
    # nzs_30: list[int] = [5, 6, 10, 15, 30]
    # Factors of 60 starting with 5
    # nzs_60: list[int] = [5, 6, 10, 12, 15, 20, 30, 60]
    # Factors of 120 starting with 5
    # nzs_120: list[int] = [5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
    # All the resolutions to use for grid in z
    nzs: list[int] = [128]

    # Run the reaction convergence study
    run_cnvg_react(G=G, nzs=nzs)

if __name__ == '__main__':
    main()
