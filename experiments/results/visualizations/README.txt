CUB-200 experiment visualizations

{
  "data_source": "experiments/results/multi_seed_summary.json",
  "seeds": [
    42,
    43,
    44
  ],
  "selection_metric": "best validation average class accuracy",
  "selected_run_history": "runs\\convnext_tiny\\fine_tuned\\seed_42\\history_convnext_tiny.json",
  "generated_files": [
    "test_accuracy_mean_sd.png",
    "test_accuracy_by_seed.png",
    "selected_run_training_curves.png"
  ]
}

Multi-seed charts show three-run means and standard deviations from official test evaluation. Training curves are from the validation-selected ConvNeXt-Tiny fine-tuned seed 42 history. Earlier production-checkpoint metrics are not mixed into these figures.
