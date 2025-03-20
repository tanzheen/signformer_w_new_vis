import torch
from collections import OrderedDict
from pprint import pprint

def inspect_pth_file(pth_path):
    """
    Load and inspect the contents of a .pth file
    
    Args:
        pth_path (str): Path to the .pth file
    """
    try:
        # Load the state dict
        state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        #state_dict = state_dict['model_state']
        # Check if it's a state dict or a complete model
        if isinstance(state_dict, OrderedDict):
            weights_dict = state_dict
        else:
            # If it's a complete model, try to get its state_dict
            weights_dict = state_dict.state_dict() if hasattr(state_dict, 'state_dict') else state_dict

        print(f"\nFile: {pth_path}")
        print("\nModel Structure:")
        print("=" * 50)
        
        # Display each layer's shape and parameters
        for key, tensor in weights_dict.items():
            print(f"\nLayer: {key}")
            print(f"Shape: {tensor.shape}")
            print(f"Data type: {tensor.dtype}")
            print(f"Number of parameters: {tensor.numel()}")
            
            # Print a small sample of values if tensor is not empty
            if tensor.numel() > 0:
                print("Sample values (first 5):")
                print(tensor.flatten()[:5])
            
            print("-" * 30)
        
        # Print total number of parameters
        total_params = sum(tensor.numel() for tensor in weights_dict.values())
        print(f"\nTotal number of parameters: {total_params:,}")
        
    except Exception as e:
        print(f"Error loading the file: {str(e)}")

def extract_and_save_visual_extractor(pth_path, save_path):
    """
    Extract visual extractor weights and save them with cleaned names
    
    Args:
        pth_path (str): Path to the source .pth file
        save_path (str): Path to save the modified weights
    """
    try:
        # Load the state dict
        state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]

        # Create new state dict with only vis_extractor parts and cleaned names
        new_state_dict = OrderedDict()
        
        for key, tensor in state_dict.items():
            if 'vis_extractor' in key:
                # Remove 'vis_extractor.' from the key
                new_key = key.replace('vis_extractor.', '')
                new_state_dict[new_key] = tensor

        # Save the new state dict
        torch.save(new_state_dict, save_path)
        print(f"Saved visual extractor weights to {save_path}")
        
        # Display the structure of saved weights
        print("\nSaved Model Structure:")
        print("=" * 50)
        for key, tensor in new_state_dict.items():
            print(f"\nLayer: {key}")
            print(f"Shape: {tensor.shape}")
            
    except Exception as e:
        print(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    # Replace with your .pth file path
    pth_file = "visual_extractor.pth"
    #pth_file = "signformer_trained_CNN.pth"
    inspect_pth_file(pth_file)
    #save_file = "visual_extractor.pth"
    #extract_and_save_visual_extractor(pth_file, save_file)
