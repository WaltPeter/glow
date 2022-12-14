from torch import nn 


class ImagePatchEmbed(nn.Module):
    TYPE = "IMAGE" 

    # Split image into patches and then embed them.
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = img_size // patch_size 

    def forward(self, x):
        b, t, c, _, _ = x.shape 
        x = x.view(b, t, c, self.n_patches, self.patch_size, self.n_patches, self.patch_size) 
        x = x.permute(0,1,3,5,4,6,2).contiguous()
        x = x.view(b, t, self.n_patches*self.n_patches, self.patch_size*self.patch_size*c) 
        # print("Embed size:", x.shape)
        return x 

    def reverse(self, x): 
        b, t, n, d = x.shape 
        x = x.view(b, t, self.n_patches, self.n_patches, self.patch_size, self.patch_size, -1) 
        c = x.shape[-1]
        x = x.permute(0,1,6,2,4,3,5).contiguous() 
        x = x.view(b, t, c, self.n_patches*self.patch_size, self.n_patches*self.patch_size) 
        return x 