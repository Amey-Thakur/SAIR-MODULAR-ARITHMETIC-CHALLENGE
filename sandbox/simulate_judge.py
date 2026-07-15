import time
import sys
import os

# Ensure submission path is available
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from submission.predict import predict_digits
except ImportError:
    print("Could not import predict_digits from submission.predict")
    sys.exit(1)

def simulate_judge():
    """Simulates the evaluation server environment."""
    print("--- SAIR Modular Arithmetic Challenge: Local Judge ---")
    
    test_batch = [
        "123*456",
        "789*101",
        "555*555",
        "999*999"
    ]
    
    print(f"Dispatching batch of {len(test_batch)} equations...")
    start_time = time.perf_counter()
    
    try:
        results = predict_digits(test_batch)
    except Exception as e:
        print(f"Submission crashed during execution: {e}")
        sys.exit(1)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Execution finished in {duration:.4f} seconds.")
    
    if len(results) != len(test_batch):
        print("FAIL: Submission did not return a result for every input.")
        sys.exit(1)
        
    print("Judge simulation successful! Output format verified.")

if __name__ == "__main__":
    simulate_judge()
