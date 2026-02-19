/**
 * @file example_multivariate.cpp
 * @brief Multivariate Analysis Sample Code
 *
 * Demonstrates the usage of covariance matrix, correlation matrix,
 * data standardization, Min-Max scaling, Principal Component Analysis (PCA),
 * and other multivariate analysis techniques.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "statcpp/multivariate.hpp"

// ============================================================================
// Helper functions for displaying results
// ============================================================================

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_subsection(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

int main() {
    std::cout << std::fixed << std::setprecision(4);

    // ============================================================================
    // 1. Covariance Matrix and Correlation Matrix
    // ============================================================================
    print_section("1. Covariance and Correlation Matrices");

    std::cout << R"(
[Concept]
Covariance matrix: Matrix summarizing covariances between multiple variables
Correlation matrix: Matrix of correlation coefficients (standardized covariances)

[Example: Health Data]
Measuring 3 variables (height, weight, age) for 8 subjects
-> Analyzing relationships between variables
)";

    // Sample data: 3 variables (height, weight, age)
    std::vector<std::vector<double>> data = {
        {170, 65, 25},
        {175, 70, 30},
        {165, 60, 22},
        {180, 75, 35},
        {172, 68, 28},
        {168, 63, 24},
        {177, 72, 32},
        {171, 67, 27}
    };

    std::cout << "\nSample data (n=" << data.size() << " subjects, p=" << data[0].size() << " variables):\n";
    std::cout << "  Height(cm), Weight(kg), Age(years)\n";

    auto cov_matrix = statcpp::covariance_matrix(data);
    auto corr_matrix = statcpp::correlation_matrix(data);

    print_subsection("Covariance Matrix");
    for (std::size_t i = 0; i < cov_matrix.size(); ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < cov_matrix[i].size(); ++j) {
            std::cout << std::setw(10) << cov_matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "-> Diagonal elements are variances, off-diagonal are covariances\n";

    print_subsection("Correlation Matrix");
    const char* var_names[] = {"Height", "Weight", "Age"};
    std::cout << "           ";
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << std::setw(10) << var_names[i];
    }
    std::cout << std::endl;

    for (std::size_t i = 0; i < corr_matrix.size(); ++i) {
        std::cout << std::setw(10) << var_names[i] << " ";
        for (std::size_t j = 0; j < corr_matrix[i].size(); ++j) {
            std::cout << std::setw(10) << corr_matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "-> Range [-1, 1], closer to 1 indicates stronger positive correlation\n";

    // ============================================================================
    // 2. Data Standardization
    // ============================================================================
    print_section("2. Data Standardization (Z-score)");

    std::cout << R"(
[Concept]
Transform each variable to mean=0, standard deviation=1
Enables comparison of variables with different scales

[Example: Variables with Different Units]
Standardize height(cm), weight(kg), age(years) to same scale
for comparison and analysis
)";

    auto standardized = statcpp::standardize(data);

    print_subsection("Original Data (first 3 rows)");
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < data[i].size(); ++j) {
            std::cout << std::setw(7) << data[i][j];
            if (j + 1 < data[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    print_subsection("Standardized Data (first 3 rows)");
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < standardized[i].size(); ++j) {
            std::cout << std::setw(7) << std::setprecision(2) << standardized[i][j];
            if (j + 1 < standardized[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << std::setprecision(4);
    std::cout << "-> Each variable is transformed to mean=0, std dev=1\n";
    std::cout << "-> Note: Sensitive to outliers\n";

    // ============================================================================
    // 3. Min-Max Scaling
    // ============================================================================
    print_section("3. Min-Max Scaling (0-1 Normalization)");

    std::cout << R"(
[Concept]
Transform each variable to min=0, max=1
Unify data range to [0,1]

[Example: Machine Learning Preprocessing]
Scale variables for neural networks and
distance-based algorithms
)";

    auto scaled = statcpp::min_max_scale(data);

    print_subsection("After Min-Max Scaling (first 3 rows)");
    for (std::size_t i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < scaled[i].size(); ++j) {
            std::cout << std::setw(7) << scaled[i][j];
            if (j + 1 < scaled[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "-> All values fall within [0, 1] range\n";
    std::cout << "-> Note: Strongly affected by outliers\n";

    // ============================================================================
    // 4. Principal Component Analysis (PCA)
    // ============================================================================
    print_section("4. Principal Component Analysis (PCA)");

    std::cout << R"(
[Concept]
Technique to summarize multivariate data into fewer principal components (linear combinations)
Finds new axes that maximize data variance

[Example: Dimensionality Reduction]
Compress 3-variable data into 2 principal components
-> For visualization and computational efficiency
)";

    std::size_t n_components = 2;
    auto pca_result = statcpp::pca(data, n_components);

    print_subsection("PCA Results (" + std::to_string(n_components) + " components)");

    std::cout << "\nPrincipal Component Loadings:\n";
    for (std::size_t i = 0; i < pca_result.components.size(); ++i) {
        std::cout << "  PC" << (i + 1) << ": [";
        for (std::size_t j = 0; j < pca_result.components[i].size(); ++j) {
            std::cout << std::setw(8) << pca_result.components[i][j];
            if (j + 1 < pca_result.components[i].size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "-> Contribution of each variable to principal components\n";

    std::cout << "\nExplained Variance:\n";
    for (std::size_t i = 0; i < pca_result.explained_variance.size(); ++i) {
        std::cout << "  PC" << (i + 1) << ": " << pca_result.explained_variance[i] << std::endl;
    }

    std::cout << "\nVariance Explained Ratio:\n";
    for (std::size_t i = 0; i < pca_result.explained_variance_ratio.size(); ++i) {
        std::cout << "  PC" << (i + 1) << ": " << (pca_result.explained_variance_ratio[i] * 100)
                  << "%" << std::endl;
    }

    double total_explained = 0.0;
    for (double ratio : pca_result.explained_variance_ratio) {
        total_explained += ratio;
    }
    std::cout << "\nCumulative Variance Ratio: " << (total_explained * 100) << "%\n";
    std::cout << "-> " << n_components << " components retain " << (total_explained * 100) << "% of total information\n";

    // ============================================================================
    // 5. PCA Transform (Dimensionality Reduction)
    // ============================================================================
    print_section("5. PCA Transform (Dimensionality Reduction)");

    std::cout << R"(
[Concept]
Transform original data into principal component space
Project high-dimensional data to lower dimensions

[Example: Dimensionality Reduction for Visualization]
Convert 3D data to 2D for scatter plot
)";

    auto transformed = statcpp::pca_transform(data, pca_result);

    print_subsection("Dimensionality Reduction Results");
    std::cout << "Original dimensions: " << data[0].size() << " variables\n";
    std::cout << "Reduced dimensions: " << transformed[0].size() << " components\n";

    std::cout << "\nTransformed data (first 5 observations):\n";
    std::cout << "        PC1         PC2\n";
    for (std::size_t i = 0; i < std::min(std::size_t(5), transformed.size()); ++i) {
        std::cout << "  ";
        for (std::size_t j = 0; j < transformed[i].size(); ++j) {
            std::cout << std::setw(11) << transformed[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "-> Original 3 variables condensed into 2 principal components\n";

    // ============================================================================
    // 6. Practical PCA Example
    // ============================================================================
    print_section("6. Practical Example: Feature Extraction and Component Selection");

    std::cout << R"(
[Concept]
Methods for determining required number of principal components
Select number of components where cumulative variance ratio exceeds target (e.g., 90%)

[Example: Multivariate Data Compression]
Select minimum components from 4-variable data
)";

    // Multi-dimensional data example
    std::vector<std::vector<double>> high_dim_data = {
        {2.5, 2.4, 3.1, 2.9},
        {0.5, 0.7, 0.9, 0.6},
        {2.2, 2.9, 2.7, 2.5},
        {1.9, 2.2, 2.4, 2.0},
        {3.1, 3.0, 3.5, 3.2},
        {2.3, 2.7, 2.8, 2.4},
        {2.0, 1.6, 1.8, 2.1},
        {1.0, 1.1, 1.3, 0.9},
        {1.5, 1.6, 1.7, 1.4},
        {1.1, 0.9, 1.0, 1.2}
    };

    std::cout << "\nOriginal data: " << high_dim_data.size() << " observations, "
              << high_dim_data[0].size() << " variables\n";

    // Determine number of components for 90%+ cumulative variance
    auto full_pca = statcpp::pca(high_dim_data, high_dim_data[0].size());

    print_subsection("Variance Ratio for Each Component");
    double cumulative_var = 0.0;
    std::size_t components_for_90 = 0;
    for (std::size_t i = 0; i < full_pca.explained_variance_ratio.size(); ++i) {
        cumulative_var += full_pca.explained_variance_ratio[i];
        std::cout << "  PC" << (i + 1) << ": "
                  << (full_pca.explained_variance_ratio[i] * 100) << "% "
                  << "(cumulative: " << (cumulative_var * 100) << "%)" << std::endl;
        if (cumulative_var >= 0.9 && components_for_90 == 0) {
            components_for_90 = i + 1;
        }
    }

    std::cout << "\nConclusion: " << components_for_90 << " components needed to capture 90% of variance\n";
    std::cout << "-> Dimensionality can be reduced from " << high_dim_data[0].size()
              << " to " << components_for_90 << "\n";

    // ============================================================================
    // 7. Correlation Analysis Interpretation
    // ============================================================================
    print_section("7. Correlation Matrix Interpretation");

    std::cout << R"(
[Concept]
Guidelines for interpreting correlation coefficient strength
Quantitatively evaluate relationships between variables

[Interpretation Criteria]
|r| < 0.3: Weak correlation
0.3 <= |r| < 0.7: Moderate correlation
|r| >= 0.7: Strong correlation
)";

    print_subsection("Relationship Analysis Between Variables");
    for (std::size_t i = 0; i < corr_matrix.size(); ++i) {
        for (std::size_t j = i + 1; j < corr_matrix[i].size(); ++j) {
            double r = corr_matrix[i][j];
            std::cout << "  " << var_names[i] << " vs " << var_names[j] << ": r = " << r;
            if (std::abs(r) >= 0.7) {
                std::cout << " (Strong correlation)";
            } else if (std::abs(r) >= 0.3) {
                std::cout << " (Moderate correlation)";
            } else {
                std::cout << " (Weak correlation)";
            }
            std::cout << std::endl;
        }
    }

    // ============================================================================
    // 8. Summary: Multivariate Analysis Guidelines
    // ============================================================================
    print_section("Summary: Multivariate Analysis Usage Guidelines");

    std::cout << R"(
[When to Use PCA]
- Too many variables (curse of dimensionality)
- Variables are correlated
- Want to visualize high-dimensional data
- Want to reduce noise
- As preprocessing for other algorithms

[Preparation Before Applying PCA]
1. Standardize features (especially when scales differ)
2. Check for outliers
3. Verify linear relationships

[How to Choose Number of Components]
- Cumulative variance ratio (e.g., 80-95%)
- Scree plot (elbow method)
- Kaiser criterion (eigenvalue > 1)
- Domain knowledge and interpretability

[Interpreting Principal Components]
- Loadings show contribution of each variable
- Components are orthogonal (uncorrelated)
- First component captures most variance

[When to Use Each Data Transformation]
+--------------+--------------------------------+
| Method       | Use Case                       |
+--------------+--------------------------------+
| Standardize  | Compare variables with         |
| (Z-score)    | different scales               |
|              | Sensitive to outliers          |
+--------------+--------------------------------+
| Min-Max      | Normalize to [0,1] range       |
| Scaling      | Neural network preprocessing   |
+--------------+--------------------------------+
| PCA          | Dimensionality reduction,      |
|              | noise removal                  |
|              | Visualization, efficiency      |
+--------------+--------------------------------+
)";

    return 0;
}
