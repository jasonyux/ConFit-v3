from reranking_clean.impl.qwen_list_prompt_reranker_numerical import QwenListPromptRerankerNum
from reranking_clean.pipeline import (
    build_parser,
    config_from_args,
    load_pipeline_data,
    print_config_summary,
    run_two_stage_pipeline,
)


def main() -> None:
    parser = build_parser(default_base_url="http://localhost:8004/v1")
    args = parser.parse_args()
    config = config_from_args(args)
    data = load_pipeline_data(config)
    print_config_summary(config, data)
    run_two_stage_pipeline(
        QwenListPromptRerankerNum,
        config,
        data,
        track_token_usage=True,
    )


if __name__ == "__main__":
    main()
