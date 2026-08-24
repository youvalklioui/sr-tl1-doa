import argparse

from experiments_assets.dataset_generator import generate_array, generate_dictionary, generate_dataset_test
from utils.assets_management_utils import set_random_seeds
from utils.framework_evaluator import evaluate_framework
import torch

import default_configs

SEED=1234


def parse_unknown_args(unknown_args):
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                val = unknown_args[i + 1]
                if val.lower() == 'true':
                    val = True
                elif val.lower() == 'false':
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                kwargs[key] = val
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    return kwargs


def main():
    
    
    parser = argparse.ArgumentParser(description="Main script to manage dataset generation, training, and testing of models.")
    subparsers = parser.add_subparsers(dest='command', help="Choose between dataset generation, training, and testing.")


    parser_run_full_experiment = subparsers.add_parser("run-full-experiment", help="Run the full experiment to reproduce the results.")
    

    parser_array = subparsers.add_parser("create-array", help="Generate an array and its dictionary.")
    parser_array.add_argument('--num_elements', type=int, default=30, help='Number of elements in the array.')
    parser_array.add_argument('--aperture', type=int, default=60, help='aperture of the array in λ/2 units.')

    parser_dictionary = subparsers.add_parser("create-dictionary", help="Generate a dictionary with gain-phase imperfections Ψ and mutual coupling Γ over an angular grid.")
    parser_dictionary.add_argument('--array_tag', type=str, help='Unique tag of the array in the arrays_manuscript.json.')
    parser_dictionary.add_argument('--dictionary_length', type=int, default=256, help='Length of the dictionary which is also the length of the angular grid. The angular grid is uniform and covers the range [-90, 90] degrees.')
    parser_dictionary.add_argument('--max_gain_deviation', type=float, default=0.3, help='Maximum normalized gain deviation with respect to unity, should be between 0 and 1.')
    parser_dictionary.add_argument('--max_phase_deviation', type=float, default=30.0, help='Maximum phase deviation in degs with respect to zero, should be between 0 deg and 90 deg.')
    parser_dictionary.add_argument('--correlation_length', type=int, default=5, help='The distance in degrees beyond which the deviation in gain/phase become strongly uncorrelated.')
    parser_dictionary.add_argument('--average_mutual_coupling', type=float, default=0.1, help='Average mutual coupling between array elements.')
    parser_dictionary.add_argument('--relative_variation_coupling', type=float, default=0.1, help='Relative variation of mutual coupling between array elements.')


    
    parser_dataset_test = subparsers.add_parser("create-dataset", help="Generate a test dataset.")
    parser_dataset_test.add_argument('--dictionary_tag', type=str, help='Path to the dictionary.')
    parser_dataset_test.add_argument('--log_noise_variance_values', type=float, nargs='+', default=[ -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0], help='List of log noise variance levels log10(σ²).')
    parser_dataset_test.add_argument('--num_vectors_per_variance', type=int, default=1000, help='Number of test measurement vectors per noise variance level.')
    parser_dataset_test.add_argument('--number_sources', type=int, default=8, help='Number of sources per test measurement vector.')
    parser_dataset_test.add_argument('--min_freq_separation_factor', type=int, default=3, help='Minimum frequency separation factor.')
 
        

    parser_metrics = subparsers.add_parser("evaluate-framework", help="Evaluate the performance of a model or iterative method.")
    parser_metrics.add_argument('--framework', type=str, default="LASSO", help="Name of the framework to evaluate the spectrum of or benchmark using a metric. Current frameworks supported: LASSO, L1L2, TL1, MCP, SR-LASSO, SR-TL1, SBL.")
    parser_metrics.add_argument('--dataset_tag', type=str, help="Unique tag of the dataset in the datasets_manifest.json.")
    parser_metrics.add_argument('--device', type=str, default='cpu', help="Device to be used for evaluating the output spectrum corresponding to a given framework configuration. The metric computation is done on CPU regardless of the device used for evaluating the framework.")

    args, unknown_args = parser.parse_known_args()


    set_random_seeds(SEED)

    if args.command == 'run-full-experiment':

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        create_array_default_args = default_configs.SETTINGS.get("create-array", {})
        array_tag = generate_array(**create_array_default_args)
        create_dictionary_default_args = default_configs.SETTINGS.get("create-dictionary", {})
        dictionary_tag = generate_dictionary(array_tag=array_tag, **create_dictionary_default_args)
        create_dataset_default_args = default_configs.SETTINGS.get("create-dataset", {})
        dataset_tag = generate_dataset_test(dictionary_tag=dictionary_tag, **create_dataset_default_args)

        for metric in ['detection_rate', 'false_alarm_rate', 'rmse']:
            for framework in default_configs.DEFAULT_FRAMEWORKS_CONFIGS.keys():
                config = default_configs.DEFAULT_FRAMEWORKS_CONFIGS[framework].copy()
                for regularizer in default_configs.DEFAULT_FRAMEWORKS_REGULARIZERS[framework]:
                    if framework != 'SBL':
                        config["regularization_parameter"] = regularizer
                    evaluate_framework(framework=framework, dataset_tag=dataset_tag, device=device, metric=metric, **config, **default_configs.DEFAULT_METRICS_CONFIGS)

    elif args.command == 'create-array':
        generate_array(args.num_elements, args.aperture)
    elif args.command == 'create-dictionary':
        generate_dictionary(args.array_tag, args.dictionary_length, args.correlation_length, args.max_gain_deviation, args.max_phase_deviation, args.average_mutual_coupling, args.relative_variation_coupling)
    elif args.command == 'create-dataset':
        generate_dataset_test(args.dictionary_tag, args.log_noise_variance_values, args.num_vectors_per_variance, args.number_sources, args.min_freq_separation_factor)
    elif args.command == 'evaluate-framework':
        eval_kwargs = parse_unknown_args(unknown_args)
        evaluate_framework(args.framework, args.dataset_tag, device=args.device, **eval_kwargs)

    

if __name__ == '__main__':
    main()