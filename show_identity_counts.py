import os
import pandas as pd
import argparse

def display_identity_counts(report_path, sort_by='Original_Count'):
    """
    Display identity counts sorted in ascending order
    
    Args:
        report_path: Path to the augmentation report CSV
        sort_by: Column to sort by ('Original_Count', 'Final_Count', etc.)
    """
    # Check if report exists
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return
    
    # Read the augmentation report
    df = pd.read_csv(report_path)
    
    # Sort by specified column
    sorted_df = df.sort_values(by=sort_by)
    
    # Print header
    print("\n" + "="*80)
    print(f"IDENTITY COUNTS SORTED BY {sort_by} (ASCENDING)")
    print("="*80)
    print(f"{'Identity':<30} {'Original Count':<15} {'Augmented Added':<20} {'Final Count':<15}")
    print("-"*80)
    
    # Print each row
    for _, row in sorted_df.iterrows():
        print(f"{row['Identity']:<30} {row['Original_Count']:<15} {row['Augmented_Added']:<20} {row['Final_Count']:<15}")
    
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display identity counts from augmentation report")
    parser.add_argument("--report", default="augmented_dataset/augmentation_report.csv",
                        help="Path to the augmentation report CSV file")
    parser.add_argument("--sort", default="Original_Count", choices=["Original_Count", "Final_Count", "Augmented_Added"],
                        help="Column to sort by")
    
    args = parser.parse_args()
    display_identity_counts(args.report, args.sort)