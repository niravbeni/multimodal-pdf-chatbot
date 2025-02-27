# Save this as debug_joblib.py in the same directory as your app
import os
import sys
import traceback
import joblib
from pprint import pprint

def deep_inspect(item, max_depth=3, current_depth=0, max_items=5):
    """Recursively inspect an object and print detailed information"""
    prefix = "  " * current_depth
    
    if current_depth >= max_depth:
        print(f"{prefix}[MAX DEPTH REACHED]")
        return
    
    try:
        item_type = type(item)
        print(f"{prefix}Type: {item_type}")
        
        if item is None:
            print(f"{prefix}Value: None")
        elif isinstance(item, (int, float, bool, str)):
            if isinstance(item, str) and len(item) > 100:
                print(f"{prefix}Value: {item[:100]}... (truncated, length: {len(item)})")
            else:
                print(f"{prefix}Value: {item}")
        elif isinstance(item, dict):
            print(f"{prefix}Dict with {len(item)} keys:")
            for i, (k, v) in enumerate(item.items()):
                if i >= max_items:
                    print(f"{prefix}  ... and {len(item) - max_items} more keys")
                    break
                print(f"{prefix}  Key: {k}")
                deep_inspect(v, max_depth, current_depth + 1, max_items)
        elif isinstance(item, (list, tuple)):
            print(f"{prefix}{type(item).__name__} with {len(item)} items:")
            for i, v in enumerate(item[:max_items]):
                if i >= max_items:
                    print(f"{prefix}  ... and {len(item) - max_items} more items")
                    break
                print(f"{prefix}  Item {i}:")
                deep_inspect(v, max_depth, current_depth + 1, max_items)
            if len(item) > max_items:
                print(f"{prefix}  ... and {len(item) - max_items} more items")
        elif hasattr(item, "__dict__"):
            print(f"{prefix}Object with attributes:")
            for i, (k, v) in enumerate(vars(item).items()):
                if i >= max_items:
                    print(f"{prefix}  ... and more attributes")
                    break
                print(f"{prefix}  Attribute: {k}")
                deep_inspect(v, max_depth, current_depth + 1, max_items)
        else:
            print(f"{prefix}Unknown type or complex object: {str(item)[:100]}")
    except Exception as e:
        print(f"{prefix}Error inspecting: {str(e)}")

def inspect_joblib_file(file_path):
    """Load and inspect a joblib file in detail"""
    print(f"\n{'='*80}")
    print(f"INSPECTING JOBLIB FILE: {file_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"File exists. Size: {file_size:.2f} MB")
    
    try:
        print("\nLoading data...")
        data = joblib.load(file_path)
        print("Data loaded successfully!")
        
        print("\nBASIC INFORMATION:")
        print(f"Type of data: {type(data)}")
        
        # Perform deep inspection of the data
        print("\nDETAILED INSPECTION:")
        deep_inspect(data)
        
        # If it's a dictionary, try to find documents
        if isinstance(data, dict):
            print("\nSEARCHING FOR DOCUMENTS:")
            for key in data:
                print(f"\nInspecting key: '{key}'")
                if key == "documents":
                    docs = data[key]
                    print(f"Found 'documents' key with {len(docs)} items!")
                    if docs:
                        print(f"First document type: {type(docs[0])}")
                        try:
                            if hasattr(docs[0], "page_content"):
                                print(f"Has page_content: {docs[0].page_content[:100]}...")
                            else:
                                print(f"First item: {str(docs[0])[:100]}...")
                        except:
                            print("Error accessing first document")
                else:
                    value = data[key]
                    print(f"  Type: {type(value)}")
                    if isinstance(value, (list, tuple)) and value:
                        print(f"  Contains {len(value)} items")
                        print(f"  First item type: {type(value[0])}")
        
        # Also try if it's a list directly
        elif isinstance(data, (list, tuple)):
            print(f"\nData is a {type(data).__name__} with {len(data)} items")
            if data:
                print(f"First item type: {type(data[0])}")
                try:
                    print(f"First item: {str(data[0])[:100]}...")
                except:
                    print("Error accessing first item")
        
    except Exception as e:
        print(f"\nERROR loading or inspecting file: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Use command line argument or default path
    file_path = sys.argv[1] if len(sys.argv) > 1 else "./preprocessed_data/primary_collection.joblib"
    inspect_joblib_file(file_path)
    
    print("\nCOMPLETE!")