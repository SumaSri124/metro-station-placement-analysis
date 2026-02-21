# Metro Station Placement Analysis (ML)

This project explores how machine learning can be used to analyze potential metro station locations in a city.

A synthetic spatial dataset was generated using population density, traffic intensity, and facility distribution as proxy indicators of activity. A suitability score was computed for each location, and KMeans clustering was used to identify candidate station zones.

An XGBoost model was then trained to analyze feature importance and understand which factors most influenced the suitability score.

## Outputs
- Predicted station coordinates  
- Station placement visualization  
- Feature importance plot  
