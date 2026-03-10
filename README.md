# MARKET-BASKET-ANALYSIS-APRIORI-ALGORITHM-
Discover which products are frequently bought TOGETHER using the Apriori algorithm. Generate cross-selling recommendations.
[README_project21.md](https://github.com/user-attachments/files/25858297/README_project21.md)
# Market Basket Analysis — Online Retail (Apriori Algorithm)

## Overview
Market basket analysis on **real UK online retail data** (500K+ transactions) using the Apriori algorithm. Discovers which products are frequently bought together, generates association rules with Support/Confidence/Lift metrics, and builds a product recommendation engine for cross-selling.

**Built by:** Nithin Kumar Kokkisa — Senior Demand Planner with 12+ years at HPCL managing 180,000 MTPA facility.

---

## Dataset
- **Source**: [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Size**: 500K+ transactions from a UK online retailer (2010-2011)
- **Products**: ~4,000 unique items
- **Customers**: ~4,300 unique customers

## Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Support** | P(X and Y) | How often X and Y appear together |
| **Confidence** | P(Y\|X) = P(X,Y)/P(X) | If someone buys X, probability they buy Y |
| **Lift** | Confidence / P(Y) | How much MORE likely Y is when X is present (>1 = positive) |

## Business Applications
- **Cross-selling**: "Customers who bought X also bought Y"
- **Product placement**: Place associated products near each other
- **Bundle pricing**: Combo deals for high-lift product pairs
- **Email marketing**: Recommend associated products to past buyers

## Tools & Technologies
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **mlxtend** (Apriori algorithm, association rules)

---

## About
Part of a **30-project data analytics portfolio**. See [GitHub profile](https://github.com/Kokkisa) for the full portfolio.
