import os
import cv2
import argparse
import numpy as np
import block_match_ncc_updated
import block_match_ssd_updated
import LKTracker
import pyramid_implementation_1
import pyramid_implementation_2

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-m', '--method_num', type=str, default='method', required=True, \
                                                        help="Number of method to be used")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="category of method to be used")
    args = parser.parse_args()
    
    return args


def Block_matching(args):
    if (args.category == "ncc"):
        block_match_ncc_updated.block_match_ncc("../data/Bolt/")
    elif (args.category == "ssd"):
        block_match_ssd_updated.block_match_ssd("../data/Bolt/")
    else:
        print("Please provide a valid category")


def LK_tracker(args):
    if (args.category == "affine"):
        LKTracker.LKTracker("../data/Bolt/", "affine")
    elif (args.category == "projective"):
        LKTracker.LKTracker("../data/Bolt/", "projective")
    elif (args.category == "translation"):
        LKTracker.LKTracker("../data/Bolt/", "translation")
    else:
        print("Please provide a valid category")


def LK_pyramid(args):
    if (args.category == "affine"):
        pyramid_implementation_2.LKpyramid_imp2("../data/Bolt/", "affine", 1)
    elif (args.category == "projective"):
        pyramid_implementation_2.LKpyramid_imp2("../data/Bolt/", "projective", 1)
    elif (args.category == "translation"):
        pyramid_implementation_2.LKpyramid_imp2("../data/Bolt/", "translation", 1)
    else:
        print("Please provide a valid category")




def main(args):
    if args.method_num not in "123":
        raise ValueError("Method number should be one of 1/2/3 - Found: %s"%args.method_num)
    FUNCTION_MAPPER = {
            "1": Block_matching,
            "2": LK_tracker,
            "3": LK_pyramid,
        }

    FUNCTION_MAPPER[args.method_num](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
