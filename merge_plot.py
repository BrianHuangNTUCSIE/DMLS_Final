from PIL import Image
import os

datasets = ['mnist', 'cifar10', 'cifar100']
models = ['LR', 'DNN', 'CNN', 'DenseNet']
types = ['IID', 'non_IID']

def merge_images(image_type):
    grid_rows = len(models)
    grid_cols = len(datasets)
    image_grid = []

    # Load images row-wise
    for model in models:
        row_images = []
        for dataset in datasets:
            path = f"{dataset}_plot/{image_type}_{model}.png"
            img = Image.open(path)
            row_images.append(img)
        image_grid.append(row_images)

    # Get size from first image
    img_width, img_height = image_grid[0][0].size
    merged_image = Image.new('RGB', (img_width * grid_cols, img_height * grid_rows))

    # Paste images into grid
    for row_idx, row_images in enumerate(image_grid):
        for col_idx, img in enumerate(row_images):
            x_offset = col_idx * img_width
            y_offset = row_idx * img_height
            merged_image.paste(img, (x_offset, y_offset))

    merged_image.save(f"{image_type}_result.png")

for t in types:
    merge_images(t)