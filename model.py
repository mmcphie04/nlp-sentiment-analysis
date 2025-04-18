import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        # Single-head self-attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection and layer norm
        # For MultiheadAttention's key_padding_mask:
        # - True means position is masked (padding)
        # - False means position is attended to (actual tokens)
        key_padding_mask = None
        if attention_mask is not None:
            # Convert attention mask to the format expected by MultiheadAttention
            # HuggingFace masks use 1 for tokens and 0 for padding
            # We need to invert this for PyTorch's MultiheadAttention
            key_padding_mask = attention_mask.eq(0).to(dtype=torch.bool)
            
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        attn_output = self.attn_dropout(attn_output)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x
    
    def reset_parameters(self):
        """Reset all parameters in this block"""
        if hasattr(self.attention, 'reset_parameters'):
            self.attention.reset_parameters()
        else:
            # Manual reset for MultiheadAttention
            for p in self.attention.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        
        # Reset feed-forward layers
        for layer in self.ff:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, ff_dim=512, num_layers=3, dropout=0.1, max_length=256):
        super().__init__()
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.block_dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
    def forward(self, input_ids, attention_mask=None):
        # Get sequence length and apply embeddings
        seq_length = input_ids.size(1)
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        pos_emb = self.position_embedding[:, :seq_length, :]
        x = x + pos_emb
        
        x = self.embedding_dropout(x)
        x = self.embedding_norm(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            x = self.block_dropout(x)
        
        # Global mean pooling (excluding padding tokens)
        if attention_mask is not None:
            # Create mask for averaging (1 for tokens, 0 for padding)
            mask = attention_mask.float().unsqueeze(-1)
            x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            x = x.mean(dim=1)
        
        # Apply classifier
        logits = self.classifier(x)
        return logits
    
    def get_component_dict(self):
        """Returns a dictionary mapping component names to model parameters"""
        components = {}
        
        # Embedding components
        components['embeddings'] = [
            self.token_embedding,
            self.position_embedding,
            self.embedding_norm
        ]
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            components[f'block{i}.attention'] = block.attention
            components[f'block{i}.norm1'] = block.norm1
            components[f'block{i}.norm2'] = block.norm2
            components[f'block{i}.ff'] = block.ff
        
        # Classifier components
        components['classifier'] = self.classifier
        
        return components
    
    def reset_components(self, component_names):
        """Reset specific components by name"""
        components = self.get_component_dict()
        
        for name in component_names:
            if name in components:
                comp = components[name]
                if isinstance(comp, list):
                    for module in comp:
                        if hasattr(module, 'reset_parameters'):
                            module.reset_parameters()
                elif hasattr(comp, 'reset_parameters'):
                    comp.reset_parameters()
                elif isinstance(comp, nn.Sequential):
                    for module in comp:
                        if hasattr(module, 'reset_parameters'):
                            module.reset_parameters()
    
    def freeze_components(self, component_names, freeze=True):
        """Freeze or unfreeze specific components by name"""
        components = self.get_component_dict()
        
        for name in component_names:
            if name in components:
                comp = components[name]
                if isinstance(comp, list):
                    for module in comp:
                        for param in module.parameters():
                            param.requires_grad = not freeze
                else:
                    for param in comp.parameters():
                        param.requires_grad = not freeze
