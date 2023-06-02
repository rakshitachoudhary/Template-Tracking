## Instructions

1. Navigate to the directory containing source code `src`
2. Open a terminal and choose the tracker corresponding to the dataset you want to use. The files `CarTracker.py`, `BoltTracker.py` and `LiquorTracker.py` correspond to datasets `BlurCar2`, `Bolt` and `Liquor` respectively. 
3. Run the chosen file using `python3 <chosen_file_name> -m <method_number> -c <method_category>`. Here `<chosen_file_name>` is the name of chosen file, `<method_number>` is the number of method you want to run, it can have values `1`, `2`, `3` which correspond to `Block-Matching`, `Lucas Kanade Tracking`, `Pyramid based Lucas Kanade Tracking` respectively. `<method_category>` is the category of the particular `<method_number>` chosen, for method `1` it can be `ssd` corresponding to sum of squared distance or `ncc` corresponding to normalised cross correlation technique. For method `2` and `3`, it can be `affine`, `projective` or `translation`.
3. To calculate IOU score, run `eval.py` using `python3 eval.py --pred_path=<path_to_result> --gt_path=<path_to_groundtruth>`

Example command for `CarTracker.py`: `python3 CarTracker.py -m 2 -c affine`

## Instructions for live demo

1. Navigate to the directory `src`
2. Open a terminal and compile the file `live_demo.py` using `python3 live_demo.py`
3. A window contaning image from live webcam will appear on your screen. Choose the top-left point of the template you want to track followed by the botton right point of the same. The algorithm will start to track this template now.
4. Press `ESC` to exit.
