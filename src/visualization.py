"""
Enhanced visualization module for fairness experiments.
Provides comprehensive plotting functions for method comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color schemes for method categories
CATEGORY_COLORS = {
    'Baseline': '#2C3E50',
    'Pre-processing': '#3498DB',
    'In-processing': '#E74C3C',
    'Post-processing': '#27AE60'
}

METHOD_MARKERS = {
    'Baseline': 'o',
    'Pre-processing': 's',
    'In-processing': '^',
    'Post-processing': 'D'
}


def plot_accuracy_fairness_tradeoff(
    results_df: pd.DataFrame,
    accuracy_col: str = 'Accuracy',
    fairness_col: str = 'DPD',
    category_col: str = 'Category',
    method_col: str = 'Method',
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Accuracy vs Fairness Trade-off',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a scatter plot showing accuracy vs fairness trade-off.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    accuracy_col : str
        Column name for accuracy metric
    fairness_col : str
        Column name for fairness metric (lower is better)
    category_col : str
        Column name for method category
    method_col : str
        Column name for method name
    figsize : tuple
        Figure size
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    categories = results_df[category_col].unique()
    
    for category in categories:
        mask = results_df[category_col] == category
        subset = results_df[mask]
        
        ax.scatter(
            subset[fairness_col],
            subset[accuracy_col],
            c=CATEGORY_COLORS.get(category, '#95A5A6'),
            marker=METHOD_MARKERS.get(category, 'o'),
            s=150,
            label=category,
            alpha=0.8,
            edgecolors='white',
            linewidths=2
        )
        
        # Add method labels
        for idx, row in subset.iterrows():
            ax.annotate(
                row[method_col],
                (row[fairness_col], row[accuracy_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
    
    ax.set_xlabel(f'{fairness_col} (lower is fairer)', fontsize=12)
    ax.set_ylabel(f'{accuracy_col}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Category', loc='best', framealpha=0.9)
    
    # Add ideal region annotation
    ax.axhline(y=results_df[accuracy_col].max(), color='gray', linestyle='--', alpha=0.3, label='Max Accuracy')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, label='Perfect Fairness')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_grouped_bar_chart(
    results_df: pd.DataFrame,
    metrics: List[str] = ['Accuracy', 'DPD', 'EOD'],
    category_col: str = 'Category',
    method_col: str = 'Method',
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Method Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create grouped bar charts for multiple metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    metrics : list
        List of metric column names
    category_col : str
        Column name for method category
    method_col : str
        Column name for method name
    figsize : tuple
        Figure size
    title : str
        Overall title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Sort by category and method
        df_sorted = results_df.sort_values([category_col, method_col])
        
        # Create colors based on category
        colors = [CATEGORY_COLORS.get(cat, '#95A5A6') for cat in df_sorted[category_col]]
        
        bars = ax.bar(
            range(len(df_sorted)),
            df_sorted[metric],
            color=colors,
            edgecolor='white',
            linewidth=0.5
        )
        
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted[method_col], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, df_sorted[metric]):
            height = bar.get_height()
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=7
            )
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='white', label=cat) 
                      for cat, color in CATEGORY_COLORS.items() 
                      if cat in results_df[category_col].values]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_radar_chart(
    results_df: pd.DataFrame,
    metrics: List[str] = ['Accuracy', 'Precision', 'Recall', 'AUC', 'Fairness'],
    method_col: str = 'Method',
    top_n: int = 6,
    figsize: Tuple[int, int] = (10, 10),
    title: str = 'Multi-Metric Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a radar chart for multi-metric comparison.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    metrics : list
        List of metric column names
    method_col : str
        Column name for method name
    top_n : int
        Number of top methods to show
    figsize : tuple
        Figure size
    title : str
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    if len(available_metrics) < 3:
        print(f"Warning: Need at least 3 metrics for radar chart. Available: {available_metrics}")
        return None
    
    # Normalize metrics to 0-1 scale
    df_normalized = results_df.copy()
    for metric in available_metrics:
        min_val = df_normalized[metric].min()
        max_val = df_normalized[metric].max()
        if max_val > min_val:
            df_normalized[metric] = (df_normalized[metric] - min_val) / (max_val - min_val)
        else:
            df_normalized[metric] = 0.5
    
    # For fairness metrics (lower is better), invert
    fairness_metrics = ['DPD', 'EOD', 'Fairness', 'Calibration_Diff']
    for metric in fairness_metrics:
        if metric in df_normalized.columns:
            df_normalized[metric] = 1 - df_normalized[metric]
    
    # Select top methods by average score
    df_normalized['avg_score'] = df_normalized[available_metrics].mean(axis=1)
    df_top = df_normalized.nlargest(top_n, 'avg_score')
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df_top)))
    
    for i, (idx, row) in enumerate(df_top.iterrows()):
        values = row[available_metrics].tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row[method_col], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_heatmap(
    results_df: pd.DataFrame,
    rows: str = 'Method',
    cols: str = 'Dataset',
    value: str = 'Accuracy',
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Performance Heatmap',
    cmap: str = 'RdYlGn',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap for method-dataset performance.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    rows : str
        Column for row labels
    cols : str
        Column for column labels
    value : str
        Column for cell values
    figsize : tuple
        Figure size
    title : str
        Chart title
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Pivot the dataframe
    pivot_df = results_df.pivot_table(index=rows, columns=cols, values=value, aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': value}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(cols, fontsize=12)
    ax.set_ylabel(rows, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    x_col: str = 'DPD',
    y_col: str = 'Accuracy',
    category_col: str = 'Category',
    method_col: str = 'Method',
    figsize: Tuple[int, int] = (12, 8),
    title: str = 'Pareto Frontier: Accuracy vs Fairness',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a Pareto frontier plot for accuracy-fairness trade-off.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    x_col : str
        Column for x-axis (fairness, lower is better)
    y_col : str
        Column for y-axis (accuracy, higher is better)
    category_col : str
        Column for method category
    method_col : str
        Column name for method name
    figsize : tuple
        Figure size
    title : str
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find Pareto optimal points
    # A point is Pareto optimal if no other point has both lower x and higher y
    df = results_df[[x_col, y_col, method_col, category_col]].dropna()
    
    pareto_mask = np.ones(len(df), dtype=bool)
    
    for i, (xi, yi) in enumerate(zip(df[x_col], df[y_col])):
        for j, (xj, yj) in enumerate(zip(df[x_col], df[y_col])):
            if i != j:
                # j dominates i if j has lower x AND higher y
                if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                    pareto_mask[i] = False
                    break
    
    pareto_points = df[pareto_mask].sort_values(x_col)
    non_pareto = df[~pareto_mask]
    
    # Plot all points with category colors
    for category in df[category_col].unique():
        mask = df[category_col] == category
        subset = df[mask]
        
        ax.scatter(
            subset[x_col],
            subset[y_col],
            c=CATEGORY_COLORS.get(category, '#95A5A6'),
            marker=METHOD_MARKERS.get(category, 'o'),
            s=150,
            label=category,
            alpha=0.7,
            edgecolors='white',
            linewidths=2
        )
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        ax.plot(
            pareto_points[x_col],
            pareto_points[y_col],
            'k--',
            linewidth=2,
            alpha=0.5,
            label='Pareto Frontier'
        )
    
    # Highlight Pareto optimal points
    ax.scatter(
        pareto_points[x_col],
        pareto_points[y_col],
        s=300,
        facecolors='none',
        edgecolors='gold',
        linewidths=3,
        label='Pareto Optimal'
    )
    
    # Add method labels for Pareto points
    for idx, row in pareto_points.iterrows():
        ax.annotate(
            row[method_col],
            (row[x_col], row[y_col]),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )
    
    ax.set_xlabel(f'{x_col} (lower is fairer)', fontsize=12)
    ax.set_ylabel(f'{y_col}', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_information_loss(
    results_df: pd.DataFrame,
    loss_cols: List[str] = ['MI_Loss', 'Accuracy_Change', 'Prediction_Change'],
    method_col: str = 'Method',
    category_col: str = 'Category',
    figsize: Tuple[int, int] = (14, 6),
    title: str = 'Information Loss Analysis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a visualization for information loss metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    loss_cols : list
        List of information loss column names
    method_col : str
        Column name for method name
    category_col : str
        Column for method category
    figsize : tuple
        Figure size
    title : str
        Chart title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Filter to available columns
    available_cols = [c for c in loss_cols if c in results_df.columns]
    
    if not available_cols:
        print("No information loss columns found in the dataframe")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by category
    df_sorted = results_df.sort_values([category_col, method_col])
    
    x = np.arange(len(df_sorted))
    width = 0.8 / len(available_cols)
    
    for i, col in enumerate(available_cols):
        offset = (i - len(available_cols) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df_sorted[col], width, label=col, alpha=0.8)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted[method_col], rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    title: str = 'Fairness Methods Summary Dashboard',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary dashboard with multiple visualizations.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with method results
    figsize : tuple
        Figure size
    title : str
        Overall title
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Accuracy-Fairness scatter
    ax1 = fig.add_subplot(gs[0, 0])
    if 'Category' in results_df.columns and 'DPD' in results_df.columns:
        for category in results_df['Category'].unique():
            mask = results_df['Category'] == category
            subset = results_df[mask]
            ax1.scatter(
                subset.get('DPD', []),
                subset.get('Accuracy', []),
                c=CATEGORY_COLORS.get(category, '#95A5A6'),
                marker=METHOD_MARKERS.get(category, 'o'),
                s=100,
                label=category,
                alpha=0.8
            )
        ax1.set_xlabel('Demographic Parity Difference', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=10)
        ax1.set_title('Accuracy vs Fairness Trade-off', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
    
    # Top right: Bar chart for accuracy by method
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Accuracy' in results_df.columns and 'Method' in results_df.columns:
        df_sorted = results_df.sort_values('Accuracy', ascending=True)
        colors = [CATEGORY_COLORS.get(cat, '#95A5A6') for cat in df_sorted.get('Category', ['Baseline']*len(df_sorted))]
        ax2.barh(df_sorted['Method'], df_sorted['Accuracy'], color=colors)
        ax2.set_xlabel('Accuracy', fontsize=10)
        ax2.set_title('Accuracy by Method', fontsize=11, fontweight='bold')
    
    # Bottom left: Fairness metrics comparison
    ax3 = fig.add_subplot(gs[1, 0])
    fairness_cols = [c for c in ['DPD', 'EOD', 'Calibration_Diff'] if c in results_df.columns]
    if fairness_cols and 'Method' in results_df.columns:
        df_melted = results_df.melt(
            id_vars=['Method'],
            value_vars=fairness_cols,
            var_name='Metric',
            value_name='Value'
        )
        x = np.arange(len(results_df['Method'].unique()))
        width = 0.25
        for i, metric in enumerate(fairness_cols):
            subset = df_melted[df_melted['Metric'] == metric]
            ax3.bar(x + i*width, subset['Value'], width, label=metric, alpha=0.8)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(results_df['Method'].unique(), rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Value (lower is better)', fontsize=10)
        ax3.set_title('Fairness Metrics Comparison', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
    
    # Bottom right: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    if 'Category' in results_df.columns:
        summary_stats = results_df.groupby('Category').agg({
            'Accuracy': ['mean', 'std'],
            'DPD': ['mean', 'std'] if 'DPD' in results_df.columns else ['count', 'count']
        }).round(3)
        
        table_data = []
        for cat in summary_stats.index:
            row = [
                cat,
                f"{summary_stats.loc[cat, ('Accuracy', 'mean')]:.3f} ± {summary_stats.loc[cat, ('Accuracy', 'std')]:.3f}",
            ]
            if 'DPD' in results_df.columns:
                row.append(f"{summary_stats.loc[cat, ('DPD', 'mean')]:.3f} ± {summary_stats.loc[cat, ('DPD', 'std')]:.3f}")
            table_data.append(row)
        
        cols = ['Category', 'Accuracy (mean ± std)']
        if 'DPD' in results_df.columns:
            cols.append('DPD (mean ± std)')
        
        table = ax4.table(
            cellText=table_data,
            colLabels=cols,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics by Category', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def format_results_table(
    results_df: pd.DataFrame,
    highlight_best: bool = True
) -> pd.DataFrame:
    """
    Format the results DataFrame for display with styling.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Raw results DataFrame
    highlight_best : bool
        Whether to highlight best values
        
    Returns
    -------
    pd.DataFrame with styled formatting
    """
    # Round numeric columns
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    df_formatted = results_df.copy()
    
    for col in numeric_cols:
        df_formatted[col] = df_formatted[col].round(4)
    
    return df_formatted
