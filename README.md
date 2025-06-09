Weather Forecasting with Machine Learning
This repository contains a project focused on improving weather forecasting using machine learning techniques. Our aim is to leverage data-driven models to provide more accurate and reliable weather predictions.

Features
- Machine Learning Models: Uses advanced ML techniques to enhance forecast accuracy.

- Data Processing: Prepares and cleans historical weather data for modeling.

- Visualization: Provides code to visualize forecast results and performance.


Technologies Used
- Python

- PyTorch / TensorFlow (depending on your setup)

- xarray (for handling gridded weather data)

- pandas, numpy (data wrangling and analysis)

- matplotlib (visualization)
# Clone the repository
```bash
git clone https://github.com/Ayoub-cmd707/Weather_forcasting.git
```
```bash
cd Weather_forcasting
```
To train the model:
```bash
  python3 Train.py --loss CRPSKERNELSTEP --ens-num 11 --target-var ssrd6 --lr 0.001 --epochs 50 --batch-size 2 --nheads 8 --num_blocks 4 --projection_channels 64 --mlp_mult 4 --num_predictors=18
```
## References
- [https://arxiv.org/abs/2412.13957](https://arxiv.org/abs/2412.13957)
- [https://arxiv.org/abs/2106.13924](https://arxiv.org/abs/2106.13924)
- [https://proceedings.neurips.cc/paper_files/paper/2022/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html](https://proceedings.neurips.cc/paper_files/paper/2022/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html)
- [https://essd.copernicus.org/articles/15/2635/2023/essd-15-2635-2023.html](https://essd.copernicus.org/articles/15/2635/2023/essd-15-2635-2023.html)
- [https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2397](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2397)
