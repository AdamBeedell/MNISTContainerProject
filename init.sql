CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    predicted_digit INT,
    confidence FLOAT,
    actual INT
);