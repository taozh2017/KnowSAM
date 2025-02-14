import albumentations as A
import cv2
import random

def build_transforms(args):
    data_transforms = {
        "train": A.Compose([
            A.OneOf([
                A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=args.image_size // 20, max_width=args.image_size // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),


        "valid_test": A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    return data_transforms


def build_weak_strong_transforms(args):
    data_transforms = {
        "train_weak":
            A.Compose([
                A.OneOf([A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST, p=1.0),], p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
                # A.OneOf([A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                #         A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                #         ], p=0.25),
                A.CoarseDropout(max_holes=8, max_height=args.image_size // 20, max_width=args.image_size // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),

        "train_strong":
            A.Compose([
                A.OneOf([A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST, p=1.0), ], p=1),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.6),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),

                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                ], p=0.5),
                A.CoarseDropout(max_holes=20, max_height=256 // 20, max_width=256 // 20,
                                min_holes=10, fill_value=0, mask_fill_value=0, p=0.7),
            ], p=1.0),


        "valid_test": A.Compose([
            A.Resize(*[args.image_size, args.image_size], interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    return data_transforms


