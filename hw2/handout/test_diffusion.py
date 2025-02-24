import os
import shutil
import glob
import unittest
import torch
try:
    from gradescope_utils.autograder_utils.decorators import weight
    from gradescope_utils.autograder_utils.files import check_submitted_files
except:
    # Decorator which does nothing
    def weight(n):
        return lambda func: lambda *args, **kwargs: func(*args, **kwargs)

# Force onto the CPU since GPU may yield slightly different outputs
device = "cpu"
final_save_dict = None
diffusion_test = None

def get_testing_state():
    global device
        
    final_save_dict = torch.load("data.pt", map_location=device)

    # Delay import of diffusion until we have moved diffusion.py 
    # into place on Gradescope.
    from diffusion import Diffusion

    # fix random seed for reproducibility
    torch.manual_seed(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    time_steps = 50
    img_size = 4
    batch_size = 32
    channels = 3

    model = final_save_dict["model"]

    diffusion_test = Diffusion(
        model,
        image_size=img_size,
        channels=channels,
        timesteps=time_steps,
    ).to(device)
    
    # Inject our custom noise_like function with fixed random seeds
    def noise_like(shape, device):
        # fix random seed for reproducibility
        torch.manual_seed(2024)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        noise = lambda: torch.randn(shape)
        noise = noise().to(device)
        return noise

    diffusion_test.noise_like = noise_like

    return final_save_dict, diffusion_test

class TestDiffusion(unittest.TestCase):

    def setUp(self):
        # If the name of the current test does not contain "01", then 
        # load the testing state into the global variables
        if "01" not in self.id():
            global final_save_dict
            global diffusion_test
            if final_save_dict is None or diffusion_test is None:
                final_save_dict, diffusion_test = get_testing_state()

    @weight(0)
    def test_01_submitted_files(self):
        """[T01] Check submitted files"""
        if os.path.exists('/autograder/submission'):
            # We are running on Gradescope
            print('Submitted files: ', end='')
            print([x.replace('/autograder/submission/', '') for x in
                glob.glob('/autograder/submission/**/*', recursive=True)])
            missing_files = check_submitted_files(['diffusion.py'])
            assert len(missing_files) == 0, f"Missing files: {missing_files}"
            shutil.copy('/autograder/submission/diffusion.py', './diffusion.py')
        
    @weight(0)
    def test_02_noise_like(self):
        for ind, curr_dict in enumerate(final_save_dict["noise_like"]):
            shape = curr_dict["shape"]
            expected_res = curr_dict["expected_res"]
            res = diffusion_test.noise_like(shape, device)
            torch.testing.assert_close(expected_res, res, atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_03_p_sample(self):
        for ind, curr_dict in enumerate(final_save_dict["p_sample"]):
            x = curr_dict["x"]
            t = curr_dict["t"]
            t_index = curr_dict["t_index"]
            expected_res = curr_dict["expected_res"]
            res = diffusion_test.p_sample(x, t, t_index)
            torch.testing.assert_close(expected_res, res, atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_04_p_losses(self):
        for ind, curr_dict in enumerate(final_save_dict["p_losses"]):
            x_start = curr_dict["x_start"]
            t = curr_dict["t"]
            noise = curr_dict["noise"]
            expected_res = curr_dict["expected_res"]
            res = diffusion_test.p_losses(x_start, t, noise)
            torch.testing.assert_close(expected_res, res, atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_05_p_sample_loop(self):
        for ind, curr_dict in enumerate(final_save_dict["p_sample_loop"]):
            img = curr_dict["img"]
            expected_res = curr_dict["expected_res"]
            res = diffusion_test.p_sample_loop(img)
            torch.testing.assert_close(expected_res, res, atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_06_sample(self):
        for ind, curr_dict in enumerate(final_save_dict["sample"]):
            b_size = curr_dict["b_size"]
            expected_res = curr_dict["expected_res"]
            res = diffusion_test.sample(b_size)
            torch.testing.assert_close(expected_res, res, atol=1e-4, rtol=1e-4)

    @weight(1)
    def test_07_q_sample(self):
        for ind, curr_dict in enumerate(final_save_dict["q_sample"]):
            x_start = curr_dict["x_start"]
            t = curr_dict["t"]
            noise = curr_dict["noise"]
            expected_res = curr_dict["expected_res"]
            res = diffusion_test.q_sample(x_start, t, noise)
            torch.testing.assert_close(expected_res, res, atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    unittest.main()
