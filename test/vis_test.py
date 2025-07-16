import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

FONT_SIZE = 25

def plot_bar_chart(df, c_name, y_name):
    # Create main figure for the bar chart
    plt.rcParams['font.size'] = FONT_SIZE
    fig, ax = plt.subplots(figsize=(12, 6))
    # Create the bar chart
    bars = ax.bar(
        x=np.arange(len(df)),
        height=df[c_name],
        tick_label=df.index + 1
    )
    
    # Axis formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    # ax.set_ylim(0, 1)
    # ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Add text annotations to the top of the bars
    bar_color = bars[0].get_facecolor()
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=bar_color,
            weight='bold'
        )

    ax.set_xlabel('Image', fontweight='bold')
    ax.set_ylabel(y_name, fontweight='bold')
    

    
    # Save and show the plots
    fig.tight_layout()
    fig.savefig(f'bar_{c_name}.png')
    plt.show()
    

def plot_test_entropy_complexity(df):
    """Plot entropy-complexity scatter plot for test metrics with row index+1 as labels"""
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.rcParams['font.size'] = FONT_SIZE
    # Create a color palette
    # colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    # Plot points
    scatter = ax.scatter(
        df['permutation_entropy'], 
        df['statistical_complexity'],
        s=100,  # marker size
        alpha=0.8,
        edgecolors='white',
        linewidths=0.5
    )
    
    # Add labels (row index + 1)
    for i, row in df.iterrows():
        label = str(i + 1)  # Row index + 1
        ax.annotate(
            label,
            (row['permutation_entropy'], row['statistical_complexity']),
            xytext=(7, 0),  # Small offset
            textcoords='offset points',
            fontweight='bold',
            ha='left',
            va='center'
        )
    
    # Add image names as hover text (this will only work in interactive mode)
    for i, row in df.iterrows():
        ax.annotate(
            row['image_name'],
            (row['permutation_entropy'], row['statistical_complexity']),
            xytext=(0, 0),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
            visible=False,
            fontweight='bold'
        )
    
    # Styling
    ax.set_xlabel('Permutation Entropy', fontweight='bold')
    ax.set_ylabel('Statistical Complexity', fontweight='bold')
    
    # Enhanced styling
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('entropy_complexity_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_test_entropy_complexity_slide(df):
    """Plot entropy-complexity scatter plot for test metrics with row index+1 as labels"""
    plt.rcParams['font.size'] = FONT_SIZE
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a color palette
    # colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    # Plot points
    scatter = ax.scatter(
        df['permutation_entropy'], 
        df['statistical_complexity'],
        s=100,  # marker size
        alpha=0.8,
        edgecolors='white',
        linewidths=0.5
    )
    
    # Add labels (row index + 1)
    for i, row in df.iterrows():
        label = str(i + 1)  # Row index + 1
        ax.annotate(
            label,
            (row['permutation_entropy'], row['statistical_complexity']),
            xytext=(7, 0),  # Small offset
            textcoords='offset points',
            fontweight='bold',
            ha='left',
            va='center'
        )
    
    # # Add image names as hover text (this will only work in interactive mode)
    # for i, row in df.iterrows():
    #     ax.annotate(
    #         row['image_name'],
    #         (row['permutation_entropy'], row['statistical_complexity']),
    #         xytext=(0, 0),
    #         textcoords='offset points',
    #         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
    #         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
    #         visible=False,
    #         fontweight='bold'
    #     )
    
    # Styling
    ax.set_xlabel('Permutation Entropy', fontweight='bold')
    ax.set_ylabel('Statistical Complexity', fontweight='bold')
    ax.set_ylim(-0.049, 0.3) 
    ax.set_xlim(-0.12, 1.12)
    
    # Enhanced styling - transparent background
    ax.set_facecolor('none')  # Transparent plot area
    fig.patch.set_facecolor('none')  # Transparent figure background
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5,color='#000000')
    
    
    plt.tight_layout()
    plt.savefig('entropy_complexity_scatter_slide.png', dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('./test_metrics_results_slide.csv')
    # plot_bar_chart(df, 'mdl_complexity', 'MDL Complexity')
    # plot_bar_chart(df, 'zip_compression_ratio', 'ZIP Compression')
    plot_test_entropy_complexity_slide(df)