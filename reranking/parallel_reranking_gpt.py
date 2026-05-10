from reranking_clean.impl.universal_list_prompt_reranker_numerical import UniversalListPromptRerankerNum
from reranking_clean.pipeline import (
    build_parser,
    config_from_args,
    load_pipeline_data,
    print_config_summary,
    run_two_stage_pipeline,
)


def main() -> None:
    parser = build_parser(default_base_url=None)
    args = parser.parse_args()
    config = config_from_args(args)
    data = load_pipeline_data(config)
    print_config_summary(config, data)
    run_two_stage_pipeline(
        UniversalListPromptRerankerNum,
        config,
        data,
        track_token_usage=False,
    )


if __name__ == "__main__":
    main()
