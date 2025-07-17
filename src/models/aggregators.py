import torch
import torch.nn as nn
import torch.nn.functional as F


def global_median_pool(x, batch, size=None):
    """
    This is a new type of pooling that doesn't exist in PyG yet.
    It is similar to global_mean_pool, but uses the median instead of the mean.

    It seems to be slightly more rebust to outliers.
    """
    return torch.cat([
        torch.median(x[batch == this_batch], dim=-2, keepdim=True).values
        for this_batch in range(batch.max().item() + 1)
    ])

def readout(x, batch, size=None):

    batch_size = batch.max().item()
    mean = torch.cat([
        torch.mean(x[batch == this_batch], dim=-2, keepdim=True)
        for this_batch in range(batch_size + 1)
    ])
    max = torch.cat([
        torch.max(x[batch == this_batch], dim=-2, keepdim=True).values
        for this_batch in range(batch_size + 1)
    ])
    return torch.cat([mean, max], dim=-1)


def batch_softmax(x, batch):
    """
    Batch-wise softmax operator.

    Args:
    - x (Tensor): [total_samples, num_heads], the input tensor.
    - batch (Tensor): [total_samples,], the batch tensor, which assigns each sample to a batch.

    Returns:
    - Tensor: [total_samples, num_heads], the softmaxed tensor.
    """
    # Find the maximum value in each batch
    x_sm = torch.zeros_like(x)
    for b in torch.unique(batch):
        x_sm[batch == b] = F.softmax(x[batch == b], dim=0)  # [num_samples_in_batch, num_heads]

    return x_sm


class EasyAttentionAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads: int, cat: bool = False):
        """
        Learns a single attention score per sample in a batch, softmaxes the scores over the batch,
        and uses the softmaxed scores to take a weighted sum of the samples in the batch.

        Args:
        - cat: bool - If True, the results from separate heads are concatenated 
                                before being passed through the final linear layer.
                                Otherwise, the results are averaged.
        """
        super(EasyAttentionAggregator, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cat = cat

        # don't use a CLS token, just learn a score per sample, and softmax the scores
        self.weights = nn.Parameter(torch.zeros(num_heads, embed_dim))  # [num_heads, embed_dim]

    def forward(self, x, batch, return_att: bool = False):
        # x shape: (total_samples, embed_dim)
        # batch shape: (total_samples,)
        n_batches = torch.unique(batch).size(0)

        att = torch.matmul(x, self.weights.T)  # [total_samples, num_heads]

        # softmax over the batch
        att_sm = batch_softmax(att, batch)  # [total_samples, num_heads]

        # weighted sum
        h = torch.zeros(n_batches, self.num_heads, self.embed_dim, device=x.device)  # [n_batches, num_heads, embed_dim]
        for b in torch.unique(batch):
            x_b = x[batch == b]
            att_b = att_sm[batch == b]  # [num_samples_in_batch, num_heads]
            h[b, :, :] = torch.matmul(att_b.T, x_b)  # [num_heads, embed_dim]

        if self.cat:
            h = h.view(n_batches, -1)
        else:
            h = h.mean(dim=1)

        if return_att:
            return h, att_sm
        return h  # [n_batches, embed_dim]

        
class SelfAttentionAggregator(nn.Module):
    def __init__(self, embed_dim, heads: int, cat: bool = False):
        super(SelfAttentionAggregator, self).__init__()
        self.cat = False  ## CAT ignored!
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1, 1, embed_dim]
        self.multihead_attn = nn.MultiheadAttention(embed_dim, heads)
        self.linear = nn.Linear(embed_dim, 1)  # Assuming binary classification or regression
    
    def forward(self, x, batch, return_att: bool = False):
        # x shape: (total_samples, embed_dim)
        # batch shape: (total_samples,)

        # print(f"x: {x.shape}")
        # print(f"batch: {batch.shape}")
        
        unique_batches = torch.unique(batch)
        batch_outputs = []
        
        # print(f"unique_batches: {unique_batches}")
        
        for b in unique_batches:
            # Select samples belonging to the current batch
            batch_mask = (batch == b)
            x_batch = x[batch_mask]  # [num_samples_in_batch, embed_dim]
            
            # Add CLS token to the input
            cls_tokens = self.cls_token.expand(x_batch.size(0), -1, -1).mean(dim=0, keepdim=True)  # [1, embed_dim]
            x_batch = torch.cat((cls_tokens, x_batch.unsqueeze(1)), dim=0)  # [num_samples_in_batch + 1, embed_dim]
            
            # Prepare input for MultiheadAttention
            # x_batch = x_batch.unsqueeze(1)  # [num_samples_in_batch + 1, 1, embed_dim]
            
            # print(x_batch.shape)

            # Self-attention mechanism
            attn_output, _ = self.multihead_attn(x_batch, x_batch, x_batch)  # TODO: I think Q can be just the cls token here.

            # Extract the CLS token output
            cls_output = attn_output[0]  # [1, embed_dim]
            batch_outputs.append(cls_output)
        
        # Stack all batch outputs
        output = torch.stack(batch_outputs).squeeze(1)  # [num_batches, embed_dim]
        
        # Pass the CLS token outputs through a linear layer
        # output = self.linear(batch_outputs)  # [num_batches, 1]

        if return_att:
            return output, None
        return output
