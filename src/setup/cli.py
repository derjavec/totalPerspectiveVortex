import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Total Perspective Vortex')
    
    parser.add_argument('--level', type=str, default='INFO', choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help='set logging level')
    parser.add_argument('--subject', type=int, default=None, help='ID of the subject to analyze (default: all subjects)')
    parser.add_argument(
        "--model", 
        type=str, 
        default="logistic", 
        choices=["logistic", "randomforest"],
        help="Choose classifier"
    )
    parser.add_argument('--transformer', type=str, default=None, choices=['pca', 'csp'], help='transformer choice (pca or csp)')
    
    return parser.parse_args()