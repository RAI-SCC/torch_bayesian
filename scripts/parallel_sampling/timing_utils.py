import torch

# Stores all durations and cumulative time per function
cuda_call_durations = {}
cuda_cumulative_times = {}

def cuda_time_function(func):
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()  # Wait for the events to complete

        elapsed = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds

        # Record individual call duration
        if func.__name__ not in cuda_call_durations:
            cuda_call_durations[func.__name__] = []
        cuda_call_durations[func.__name__].append(elapsed)

        # Update cumulative time
        cuda_cumulative_times[func.__name__] = cuda_cumulative_times.get(func.__name__, 0.0) + elapsed

        return result
    return wrapper

def print_cuda_timing_summary():
    print("\nCUDA Function Timing Summary:")
    for name in cuda_call_durations:
        total = cuda_cumulative_times[name]
        calls = cuda_call_durations[name]
        avg = total / len(calls)
        print(f"Function '{name}':")
        print(f"  Total time: {total:.3f} s over {len(calls)} calls")
        print(f"  Avg time per call: {avg:.4f} s")
        print(f"  Individual calls: {[f'{t:.4f}' for t in calls]}")
