import sys
import os
import torch
import unittest
import importlib
import glob
import shutil
try:
    from gradescope_utils.autograder_utils.decorators import weight
    from gradescope_utils.autograder_utils.files import check_submitted_files
except:
    # Decorator which does nothing
    def weight(n):
        return lambda func: lambda *args, **kwargs: func(*args, **kwargs)

from transformers import CLIPTokenizer

def delayed_import(module_name, global_var_name=None):
    '''Import a module later in the code (e.g. so that a package file can be moved into place)
    Parameters:
        module_name (str): The name of the module to import
        global_var_name (str): [Optional] The name to use for the imported module in the global namespace
    '''
    if global_var_name is None:
        global_var_name = module_name
    # Use importlib.import_module to handle both top-level modules and submodules
    globals()[global_var_name] = importlib.import_module(module_name)
    
def delayed_import_function_or_class(module_name, object_name, global_var_name=None):
    '''Import a specific function or class from a module later in the code.
    Parameters:
        module_name (str): The name of the module that contains the function or class to import
        object_name (str): The name of the function or class to import
        global_var_name (str): [Optional] The name to use for the imported object in the global namespace
    '''
    if global_var_name is None:
        global_var_name = object_name
    # Import the module
    module = importlib.import_module(module_name)    
    # Get the specific function or class from the module
    globals()[global_var_name] = getattr(module, object_name)

def delayed_imports():
    # This has the effect of importing the modules as if we had called:
    # from prompt2prompt import get_replacement_mapper_, MyLDMPipeline, MySharedAttentionSwapper
    delayed_import('prompt2prompt')
    delayed_import_function_or_class('prompt2prompt', 'get_replacement_mapper_')
    delayed_import_function_or_class('prompt2prompt', 'MyLDMPipeline')
    delayed_import_function_or_class('prompt2prompt', 'MySharedAttentionSwapper')

    # After delayed import, set the device and dtype for prompt2prompt
    prompt2prompt.torch_device = "cpu"
    prompt2prompt.torch_dtype = torch.float32

testcases_ref_dict_path = "data.pt"

class TestP2P(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestP2P, self).__init__(*args, **kwargs)
        self.is_setup = False
    
    def setUp(self):
        if 'prompt2prompt' in sys.modules and self.is_setup == False:
            self.testcases_ref_mappers = (
                torch.load(testcases_ref_dict_path, weights_only=True)['replacement_mapper'])
            self.testcases_ref_swapper = (
                torch.load(testcases_ref_dict_path, weights_only=True)['swapper'])
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="tokenizer",
                torch_dtype=prompt2prompt.torch_dtype,
            )
            self.is_setup = True
        
    @weight(1)
    def test_01_submitted_files(self):
        """[T01] Check submitted files"""
        if os.path.exists('/autograder/submission'):
            # We are running on Gradescope
            print('Submitted files: ', end='')
            print([x.replace('/autograder/submission/', '') for x in
                glob.glob('/autograder/submission/**/*', recursive=True)])
            required_files = ['prompt2prompt.py']
            missing_files = check_submitted_files(required_files)
            assert len(missing_files) == 0, f"Missing files: {missing_files}"
            for file in required_files:
                shutil.copy(f'/autograder/submission/{file}', f'./{file}')
        delayed_imports()

    # ----------------------------------------------------------------
    # Tests of get_replacement_mapper_
    # ----------------------------------------------------------------

    def check_mapper(self, x, y, test_name):
        mapper = get_replacement_mapper_(x, y, self.tokenizer)
        reference_mapper = self.testcases_ref_mappers[test_name]
        torch.testing.assert_close(mapper, reference_mapper, rtol=1e-4, atol=1e-4)

    @weight(1)
    def test_02_mapper_one_to_one(self):
        x = "A lion eating a burger"
        y = "A seal eating a burger"
        self.check_mapper(x, y, 'one_to_one')

    @weight(1)
    def test_03_mapper_one_to_n(self):
        x = "A lion eating a burger"
        y = "A hippopotamus eating a burger"
        self.check_mapper(x, y, 'one_to_n')

    @weight(1)
    def test_04_mapper_n_to_one(self):
        x = "A matchstick falling"
        y = "A ball falling"
        self.check_mapper(x, y, 'n_to_one')

    @weight(1)
    def test_05_mapper_n_to_m(self):
        x = "A house with unimaginably large windows"
        y = "A house with inconceivably large windows"
        self.check_mapper(x, y, 'n_to_m')

    @weight(1)
    def test_06_mapper_n_to_n(self):
        x = "A house with unimaginably large windows"
        y = "A house with unrealistically large windows"
        self.check_mapper(x, y, 'n_to_n')

    # ----------------------------------------------------------------
    # Tests of MyLDMPipeline.get_random_noise
    # ----------------------------------------------------------------

    @weight(1)
    def test_07_get_random_noise_same_noise(self):
        batch_size, channel, height, width = 4, 3, 64, 64
        generator = torch.Generator().manual_seed(42)  # Setting a seed for reproducibility

        # Run the function with same_noise_in_batch=True
        latents = MyLDMPipeline.get_random_noise(batch_size, channel, height, width, generator, same_noise_in_batch=True)

        # Check the shape
        assert latents.shape == (batch_size, channel, height, width), "Output tensor has incorrect shape."

        # Check if all entries in the batch are the same
        for i in range(1, batch_size):
            assert torch.equal(latents[0], latents[i]), "Not all entries in the batch are the same."

    @weight(1)
    def test_08_get_random_noise_different_noise(self):
        batch_size, channel, height, width = 4, 3, 64, 64
        generator = torch.Generator().manual_seed(42)  # Setting a seed for reproducibility

        # Run the function with same_noise_in_batch=False
        latents = MyLDMPipeline.get_random_noise(batch_size, channel, height, width, generator, same_noise_in_batch=False)

        # Check the shape
        assert latents.shape == (batch_size, channel, height, width), "Output tensor has incorrect shape."

        # Check if entries in the batch are different
        all_unique = all(not torch.equal(latents[i], latents[j]) for i in range(batch_size) for j in range(i + 1, batch_size))
        assert all_unique, "Entries in the batch are not unique when they should be."


    # ----------------------------------------------------------------
    # Tests of MySharedAttentionSwapper
    # ----------------------------------------------------------------

    def check_swapper(self, prompts, test_name):
        swapper = MySharedAttentionSwapper(prompts, self.tokenizer, 1.0, 0.0)
        swapper.cur_step = 0
        swapper.num_steps_cross = 10
        test_case = self.testcases_ref_swapper[test_name]
        result = swapper.swap_attention_probs(test_case, True)
        reference_result = self.testcases_ref_swapper[test_name + '_result']
        torch.testing.assert_close(result, reference_result, rtol=1e-4, atol=1e-4)
        
    @weight(1)
    def test_09_swapper_one_to_one(self):
        prompts = ["A lion eating a burger", "A seal eating a burger"]
        self.check_swapper(prompts, 'one_to_one')

    @weight(1)
    def test_10_swapper_one_to_n(self):
        prompts = ["A lion eating a burger", "A hippopotamus eating a burger"]
        self.check_swapper(prompts, 'one_to_n')

    @weight(1)
    def test_11_swapper_n_to_one(self):
        prompts = ["A matchstick falling", "A ball falling"]
        self.check_swapper(prompts, 'n_to_one')

    @weight(1)
    def test_12_swapper_n_to_m(self):
        prompts = ["A house with unimaginably large windows", "A house with inconceivably large windows"]
        self.check_swapper(prompts, 'n_to_m')

    @weight(1)
    def test_13_swapper_n_to_n(self):
        prompts = ["A house with unimaginably large windows", "A house with unrealistically large windows"]
        self.check_swapper(prompts, 'n_to_n')



if __name__ == "__main__":
    unittest.main()