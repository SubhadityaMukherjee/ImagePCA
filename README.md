# Image Clustering Visualization for large datasets

## Instructions
- Install all the requirements. ```sh pip install -r requirements.txt```
- Modify the folder path and the type of image you are looking for in the script 
- k chooses a subset of your entire data for testing purposes. Change it to None if you want to run it for your whole data
- image_size is in pixels. Your images will be resized. I would recommend leaving it as it is. But if you do not get the results you want, you can increase it. Note that this will need more compute
- iters defines the number of iterations TSNE will run for. Increase or decrease if you do not get the results you want.
- nclust : Set it to the number of labels you want. Or None if you do not want to cluster

## Modifying
- df gives the output of clustering. It is a pandas Dataframe so do whatever with it
- If you want to run this on image crops. You can replace the preprocess function with the crop you want. Uncomment the preprocess function and the resize one in main.py
- Similarly for any other kind of preprocessing
