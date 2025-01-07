# General Deep Convergent Plug-and-Play Image Restoration Based on Primal-Dual Splitting  
Yodai Suzuki, Ryosuke Isono, and Shunsuke Ono  

MDI Lab, Institute of Science, Tokyo, Japan  

[Link to the conference proceeding](https://ieeexplore.ieee.org/document/10448023)  

## Abstract  
We propose a general deep plug-and-play (PnP) algorithm with a theoretical convergence guarantee. PnP strategies have demonstrated outstanding performance in various image restoration tasks by exploiting the powerful priors underlying Gaussian denoisers. However, existing PnP methods often lack theoretical convergence guarantees under realistic assumptions due to their ad-hoc nature, resulting in inconsistent behavior. Moreover, even when convergence guarantees are provided, they are typically designed for specific settings or require a considerable computational cost in handling non-quadratic data-fidelity terms and additional constraints, which are key components in many image restoration scenarios. To tackle these challenges, we integrate the PnP paradigm with primal-dual splitting (PDS), an efficient proximal splitting methodology for solving a wide range of convex optimization problems, and develop a general convergent PnP framework. Specifically, we establish theoretical conditions for the convergence of the proposed PnP algorithm under a reasonable assumption. Furthermore, we show that the problem solved by the proposed PnP algorithm is not a standard convex optimization problem but a more general monotone inclusion problem, where we provide a mathematical representation of the solution set. Our approach efficiently handles a broad class of image restoration problems with guaranteed theoretical convergence. Numerical experiments on specific image restoration tasks validate the practicality and effectiveness of our theoretical results.

## Getting Started  

1. **Install Requirements**:  
   Use the `requirements.txt` file to install all necessary dependencies via `pip`.  

   ```bash  
   pip install -r requirements.txt  
   ```  

2. **Create Configuration File**:  
   - Update `config/setup.json` as needed for your environment.  

   **Example `setup.json`:**  

   ```json  
   {  
       "path_test": "/Users/xxx/",  
       "path_result": "/Users/xxx/",  
       "file_ext": "*.png",  
       "root_folder": "/Users/xxx/"  
   }  
   ```  

   **Fields**:  
   - `path_test`: Folder containing images to be restored.  
   - `path_result`: Folder where the experimental results will be saved.  
   - `file_ext`: File pattern for test images. This is case-sensitive.  
   - `root_folder`: Root directory where this project resides.  

3. **Set the Models**:  
   - Place the pretrained model in the `./nn` folder.  
   - In `main_gaussian.py` or `main_poisson.py`, set the filename to the variable `architecture`.  
   - The `.pth` files for the denoisers used in the proposed method are distributed [here](https://github.com/basp-group/PnP-MMO-imaging).  

4. **Test Run**:  
   - After setup, run `main_gaussian.py` or `main_poisson.py`.  
   - You can modify parameter values directly in each Python file.  

---  

## Citation  
```  
@INPROCEEDINGS{10448023,  
   author={Suzuki, Yodai and Isono, Ryosuke and Ono, Shunsuke},  
   booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},   
   title={A Convergent Primal-Dual Deep Plug-and-Play Algorithm for Constrained Image Restoration},   
   year={2024},  
   volume={},  
   number={},  
   pages={9541-9545},  
   keywords={Signal processing algorithms;Artificial neural networks;Signal processing;Robustness;Acoustics;Image restoration;Task analysis;Image restoration;plug and play (PnP);primal-dual splitting (PDS)},  
   doi={10.1109/ICASSP48485.2024.10448023}  
}  
```  