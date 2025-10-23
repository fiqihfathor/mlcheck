# mlcheck

⚡ **Lightning-fast ML dataset validation in Rust**

Catch data quality issues before training your models. mlcheck is a 
command-line tool designed specifically for ML engineers who need to 
validate datasets quickly and thoroughly.

## Why mlcheck?

Training on bad data wastes time and compute. mlcheck helps you:

- 🎯 **Validate before training** - Detect missing values, outliers, 
  class imbalance, and data quality issues
- 🔄 **Catch train/test drift** - Ensure your test set matches your 
  training distribution
- 📊 **Get actionable insights** - Receive ML-specific recommendations, 
  not just raw statistics
- ⚡ **Move fast** - Process millions of rows in seconds (10-100x 
  faster than pandas)
- 🦀 **Zero dependencies** - Single Rust binary, no Python required
