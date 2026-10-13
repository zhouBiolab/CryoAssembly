'Command-line entry point for CryoAssembly.'

import argparse
import os
import sys



def build_parser():
    'Build the command-line argument parser.'
    parser = argparse.ArgumentParser(
        prog="main.py",
        description='protassem assembly pipeline: voxelization -> point sampling -> PARENet fitting -> assembly -> refinement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Positional arguments: <data_dir>, or <density.mrc> <struct_dir> <resolution> <contour> [output_dir]')
    parser.add_argument("paths", nargs="*", metavar="PATH",
                        help='Data directory; or map, structure directory, resolution, contour and optional output directory')
    parser.add_argument("--log", action="store_true",
                        help='Write a log to the output directory (pipeline_<timestamp>.log)')
    parser.add_argument("--log-file", metavar="PATH",
                        help='Explicit log path (takes precedence over --log)')
    parser.add_argument("--chain-threshold", type=float, default=0.45, metavar="CC")
    parser.add_argument("--domain-threshold", type=float, default=0.45, metavar="CC")
    parser.add_argument("--domain-min-cc", type=float, default=0.35, metavar="CC")
    parser.add_argument("--complex-threshold", type=float, default=0.35, metavar="CC")
    parser.add_argument("--complex-min-cc", type=float, default=0.25, metavar="CC")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, metavar="TM")
    parser.add_argument("--refine-tm", type=float, default=0.75, metavar="TM")
    parser.add_argument("--num-processes", type=int, default=8, metavar="N",
                        help='Worker processes for CC scoring and local optimization')
    parser.add_argument("--batch-size", type=int, default=5, metavar="N",
                        help='Candidates per evaluation batch (each batch is locally optimized immediately)')
    parser.add_argument("--mask-radius-factor", type=float, default=1.35,
                        metavar="F", help='PARENet mask radius as a multiple of the radius of gyration')
    parser.add_argument("--min-point-distance-factor", type=float, default=0.32,
                        metavar="F", help='Minimum mask-center spacing as a multiple of the mask radius')
    parser.add_argument("--complex-domain-opt", action="store_true",
                        help='Complex domain optimization: split chains -> partition domains -> optimize domains -> merge and compare CC')
    parser.add_argument("--no-improve-accepted", dest="improve_accepted",
                        action="store_false",
                        help='Disable per-domain refinement of accepted chains (enabled by default)')
    parser.add_argument("--no-domain-opt", dest="domain_opt", action="store_false",
                        help='Disable domain optimization fallback for failed chains (enabled by default)')
    parser.add_argument("--save-all-attempts", action="store_true",
                        help='Save all fitting attempts to all_attempts/ for debugging')
    parser.add_argument("--cleanup", action="store_true",
                        help='Remove temporary work/ directories after completion')
    parser.add_argument("--no-refine", dest="do_refine", action="store_false",
                        help='Disable Step 4 homologous-domain refinement (enabled by default)')
    parser.add_argument("--homo-chain-refine", action="store_true",
                        help='Step 5 homologous-chain refinement: preserve residues, restore domains and repair weak chains using good templates')
    parser.add_argument("--no-pre-screen", dest="pre_screen", action="store_false",
                        help='Disable initial-pose parallel pre-screening (enabled by default)')
    parser.add_argument("--pre-screen-by-domain", action="store_true",
                        help='Pre-screen by domain; chains with unmatched domains continue through regular fitting')
    parser.add_argument("--no-domain-split", metavar="IDS", default="",
                        help='Chain IDs excluded from domain splitting, comma-separated, e.g. A,B')
    parser.add_argument("--runtime-config", metavar="JSON", default=None,
                        help='Runtime configuration JSON (threads, seed, etc.; omitted keys use defaults)')
    parser.add_argument("--hypothesis-chunk", type=int, default=None, metavar="N",
                        help='Hypothesis scoring chunk size (default 64; 0 disables chunks; overrides runtime-config)')
    return parser


def parse_args(argv, parser=None):
    'Parse positional inputs and command-line options.'
    parser = parser or build_parser()
    args = parser.parse_intermixed_args(argv)
    if not args.paths:
        parser.print_help()
        return parser, None
    if len(args.paths) not in (1, 4, 5):
        parser.error("expected 1 or 4-5 positional arguments, got %d"
                     % len(args.paths))
    return parser, args


def assembly_kwargs_from(args):
    return {
        "chain_threshold": args.chain_threshold,
        "initial_domain_threshold": args.domain_threshold,
        "min_domain_threshold": args.domain_min_cc,
        "similarity_threshold": args.similarity_threshold,
        "complex_threshold": args.complex_threshold,
        "complex_domain_opt": args.complex_domain_opt,
        "improve_accepted": args.improve_accepted,
        "domain_opt": args.domain_opt,
        "save_all_attempts": args.save_all_attempts,
        "cleanup": args.cleanup,
        "do_refine": args.do_refine,
        "complex_min_cc": args.complex_min_cc,
        "refine_tm": args.refine_tm,
        "num_processes": args.num_processes,
        "batch_size": args.batch_size,
        "mask_radius_factor": args.mask_radius_factor,
        "min_point_distance_factor": args.min_point_distance_factor,
        "homo_chain_refine": args.homo_chain_refine,
        "pre_screen": args.pre_screen,
        "pre_screen_by_domain": args.pre_screen_by_domain,
        "no_domain_split_chains": frozenset(
            item.strip() for item in args.no_domain_split.split(",") if item.strip()),
    }


def resolve_log_file(args):
    if args.log_file:
        return args.log_file
    if args.log:
        return True
    return None


def resolve_runtime_config(args, parser):
    """CLI override > explicit JSON > RuntimeConfig defaults."""
    from dataclasses import replace
    from protassem.runtime.config import RuntimeConfig
    config = (RuntimeConfig.from_json(args.runtime_config)
              if args.runtime_config else RuntimeConfig())
    if args.hypothesis_chunk is not None:
        if args.hypothesis_chunk < 0:
            parser.error("--hypothesis-chunk must be non-negative")
        config = replace(config, hypothesis_chunk=args.hypothesis_chunk)
    return config


def main(argv=None):
    parser, args = parse_args(sys.argv[1:] if argv is None else argv)
    if args is None:
        return 0

    runtime_config = resolve_runtime_config(args, parser)
    runtime_config.apply_thread_env()

    from protassem.core.io import find_files
    from protassem.pipeline import run_pipeline

    kwargs = assembly_kwargs_from(args)
    log_file = resolve_log_file(args)

    if len(args.paths) == 1:
        data_dir = args.paths[0]
        mrcs = find_files(data_dir, ".mrc")
        if not mrcs:
            parser.error("%s: no .mrc file found" % data_dir)
        run_pipeline(
            density_mrc=mrcs[0],
            structure_files=find_files(data_dir, ".pdb", ".cif"),
            resolution=_read_param(parser, os.path.join(data_dir, "resolution.txt")),
            contour=_read_param(parser, os.path.join(data_dir, "contour_level.txt")),
            log_file=log_file,
            assembly_kwargs=kwargs,
            runtime_config=runtime_config,
        )
        return 0

    density_mrc, struct_dir = args.paths[0], args.paths[1]
    try:
        resolution = float(args.paths[2])
        contour = float(args.paths[3])
    except ValueError as exc:
        parser.error("resolution and contour must be numbers: %s" % exc)
    run_pipeline(
        density_mrc=density_mrc,
        structure_files=find_files(struct_dir, ".pdb", ".cif"),
        resolution=resolution,
        contour=contour,
        output_dir=args.paths[4] if len(args.paths) == 5 else None,
        log_file=log_file,
        assembly_kwargs=kwargs,
        runtime_config=runtime_config,
    )
    return 0


def _read_param(parser, path):
    from protassem.core.io import read_param_file
    if not os.path.isfile(path):
        parser.error("missing parameter file: %s" % path)
    return read_param_file(path)


if __name__ == "__main__":
    raise SystemExit(main())
