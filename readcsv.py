import pandas as pd
import sys

def calculate_success_stats(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter only successful attempts (where success is True)
    successful = df[df['success'] == True]

    # Calculate sum of steps (adding 1 to each value)
    total_steps = successful['steps'].sum() + len(successful)

    # Calculate average
    average_steps = total_steps / len(successful)

    return total_steps, average_steps

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        total, average = calculate_success_stats(csv_path)
        print(f"Total steps (with +1): {total}")
        print(f"Average steps (with +1): {average:.2f}")
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
