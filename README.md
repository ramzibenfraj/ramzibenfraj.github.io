<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Car Price Prediction</title>
    <!-- Bootstrap CSS CDN - Replace with the actual Bootstrap CDN link -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <!-- Font Awesome CDN - Include the link to Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        /* Custom CSS styles */
        /* Add your custom CSS styles here */
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-md navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/plot">Dashboard</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarCollapse" aria-controls="navbarCollapse" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarCollapse">
                <ul class="navbar-nav ml-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Make Prediction</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container mt-5">
        <h1 class="text-center mb-4">
            Car Price Prediction
            <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo" height="100">
        </h1>

        <form action="/predict" method="post">
            <div class="form-group">
                <label for="hp"><i class="fas fa-tachometer-alt"></i> Horsepower (60-500):</label>
                <input type="number" class="form-control" id="hp" name="hp" min="60" max="500" required>
                <small class="text-muted">Enter a value between 60 and 500.</small>
            </div>

            <div class="form-group">
                <label for="cylinders"><i class="fas fa-cogs"></i> Cylinders:</label>
                <select class="form-control" id="cylinders" name="cylinders" required>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="6">6</option>
                    <option value="8">8</option>
                </select>
            </div>

            <div class="form-group">
                <label for="year"><i class="far fa-calendar-alt"></i> Year:</label>
                <input type="number" class="form-control" id="year" name="year" required>
            </div>

            <div class="form-group">
                <label for="doors"><i class="fas fa-door-open"></i> Doors:</label>
                <select class="form-control" id="doors" name="doors" required>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                </select>
            </div>

            <button type="submit" class="btn btn-primary"><i class="fas fa-dollar-sign"></i> Predict Price</button>
            <a href="/plot" class="btn btn-secondary ml-2"><i class="fas fa-chart-bar"></i> Dashboard</a>
        </form>

        <div id="prediction-result" class="mt-3">
            <!-- Prediction result will be displayed here -->
        </div>
    </div>

    <!-- Bootstrap JS and jQuery CDN - Optional, include if needed -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
