#%% 
import matplotlib.pyplot as plt

def display_save_image_with_mask(image, mask, save=False, display=True, save_name=None):
    try:
        image = image.permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy()
    except TypeError:
        image = image.cpu().permute(1, 2, 0).numpy()
        mask = mask.cpu().squeeze().numpy()

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    # Display the mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask) #, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    plt.tight_layout()
    if save:  
        plt.savefig(save_name)
    if display:  
        plt.show()


def display_save_image_mask_overlaid(image, mask, alpha=0.3, save=False, display=False, save_name=None):
    try:
        image = image.permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy()
    except TypeError:
        image = image.cpu().permute(1, 2, 0).numpy()
        mask = mask.cpu().squeeze().numpy()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    cmap = plt.get_cmap('jet')
    mask_colored = cmap(mask / mask.max()) 
    ax.imshow(mask_colored, alpha=alpha)
    if save:  
        plt.savefig(save_name)
    if display:  
        plt.show()