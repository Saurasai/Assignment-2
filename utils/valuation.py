def classify_price(predicted_price, actual_price, threshold=0.1):
    """
    Classify a property as Underpriced, Fairly Priced, or Overpriced.
    threshold is the tolerance (e.g. 10%)
    """
    if actual_price > predicted_price * (1 + threshold):
        return "Overpriced"
    elif actual_price < predicted_price * (1 - threshold):
        return "Underpriced"
    else:
        return "Fairly Priced"
 
def validate_price(features, predicted_price):
    """
    Validate the predicted price based on simple rules:
    e.g., price per sqft within reasonable range.
    """
    size = features.get('size')
    if size == 0 or size is None:
        return 'Invalid size'

    price_per_sqft = predicted_price / size

    # Define hypothetical realistic price per sqft range in Dubai
    min_ppsqft = 500  # example lower bound
    max_ppsqft = 5000  # example upper bound

    if price_per_sqft < min_ppsqft:
        return 'Predicted price unusually low'
    elif price_per_sqft > max_ppsqft:
        return 'Predicted price unusually high'
    else:
        return 'Price looks reasonable'
