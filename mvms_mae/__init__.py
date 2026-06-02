# MVMS-MAE: Multi-View Multi-Scale + Masked Autoencoder
# Extends MVMS contrastive learning with lightweight MAE reconstruction regularization
#
# Key innovation:
#   - Contrastive learning learns semantic invariance (ScanObjectNN robustness)
#   - MAE reconstruction preserves local geometric structure (ModelNet40 accuracy)
#   - Combined approach targets: ModelNet40 92.3-92.7%, ScanObjectNN 88%+
