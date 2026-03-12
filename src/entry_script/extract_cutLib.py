from extract_cutlib.cli import parse_args


def main() -> None:
    cfg = parse_args()
    from extract_cutlib.runner import run

    run(cfg)


if __name__ == "__main__":
    main()
