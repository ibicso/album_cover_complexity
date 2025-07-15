import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import os

FONT_SIZE = 18

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


# def plot_entropy_complexity(df):
#     """Plot 3: Total mean for each genre with non-overlapping labels"""
#     df_genres = process_genres(df.copy())
    
#     # Calculate overall means, std, and count by genre for SEM calculation
#     genre_stats = df_genres.groupby('genre')[['permutation_entropy', 'statistical_complexity']].agg(['mean', 'std', 'count']).reset_index()
#     genre_stats.columns = ['genre', 'entropy_mean', 'entropy_std', 'entropy_count', 'complexity_mean', 'complexity_std', 'complexity_count']
    
#     # Filter genres with sufficient data
#     min_count = 50
#     genre_stats = genre_stats[genre_stats['entropy_count'] >= min_count]
    
#     # Apply custom genre ordering
#     genre_stats = order_genres_custom(genre_stats)
    
#     # Calculate SEM
#     genre_stats['entropy_sem'] = genre_stats['entropy_std'] / np.sqrt(genre_stats['entropy_count'])
#     genre_stats['complexity_sem'] = genre_stats['complexity_std'] / np.sqrt(genre_stats['complexity_count'])
    
#     fig, ax = plt.subplots(figsize=(16, 12))  # Increased figure size
    
#     # Sort by custom genre order instead of count
#     genre_stats = genre_stats.sort_values(by='genre_order')
    
#     # Create a consistent color palette based on the ordered genres
#     colors = plt.cm.tab20(np.linspace(0, 1, len(genre_stats)))
    
#     # Plot with error bars
#     scatter_points = []
#     for i, (_, row) in enumerate(genre_stats.iterrows()):
#         point = ax.errorbar(row['entropy_mean'], 
#                            row['complexity_mean'],
#                            xerr=row['entropy_sem'],
#                            yerr=row['complexity_sem'],
#                            fmt='o', 
#                            markersize=12,
#                            capsize=2,
#                            capthick=0.5,
#                            elinewidth=0.5,
#                            alpha=0.8,
#                            color=colors[i],
#                            markeredgecolor='white',
#                            markeredgewidth=0.5)
#         scatter_points.append(point)
    
#     # Create text annotations for adjustText
#     texts = []
#     for i, (_, row) in enumerate(genre_stats.iterrows()):
#         # Vary annotation positions to avoid overlap
#         angle = i * (360 / len(genre_stats)) 
#         if(row['genre'] == 'R&B'): # Distribute around circle
#             angle = 290
#         elif(row['genre'] == 'Pop'):
#             angle = 120
#         elif(row['genre'] == 'Speciality'):
#             angle = 340
#         elif(row['genre'] == 'Jazz & Blues'):
#             angle = 120
#         elif(row['genre'] == 'World music'):
#             angle = 350
#         elif(row['genre'] == 'Country & Folk'):
#             angle = 290
#         elif(row['genre'] == 'Rock'):
#             angle = 250
#         elif(row['genre'] == 'Classical'):
#             angle = 120

#         offset_x = 40 * np.cos(np.radians(angle))
#         offset_y = 40 * np.sin(np.radians(angle))
        
#         text = ax.annotate(row['genre'], 
#                           (row['entropy_mean'], row['complexity_mean']),
#                           xytext=(offset_x, offset_y), textcoords='offset points',
#                           fontsize=10, fontweight='bold',
#                           ha='center', va='center',
#                           arrowprops=dict(arrowstyle='-', color=colors[i], alpha=0.6, lw=0.5))
#         texts.append(text)
    
#     # Use adjustText to prevent overlapping labels
#     # adjust_text(texts, 
#     #             arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=1, 
#     #                            shrinkA=5, shrinkB=5),  # Prevent arrows from striking through text
#     #             expand_points=(10, 10),  # Expand around points
#     #             expand_text=(1.2, 1.2),   # Expand around text
#     #             force_points=10,         # Force away from points
#     #             force_text=0.8,           # Force text away from each other
#     #             ax=ax)
    
    
#     ax.set_xlabel('Permutation Entropy', fontweight='bold', fontsize=14)
#     ax.set_ylabel('Statistical Complexity', fontweight='bold', fontsize=14)
#     ax.set_title('Musical Genre in Entropy-Complexity Space\n(Mean Values ± SEM)', 
#                 fontweight='bold', fontsize=16, pad=25)
    
#     ax.set_ylim(0.1125, 0.14)
#     ax.set_xlim(0.82, 0.87)
    
#     # Enhanced styling
#     ax.set_facecolor('#fafafa')
#     ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
#     # Add subtle border
#     for spine in ax.spines.values():
#         spine.set_edgecolor('#cccccc')
#         spine.set_linewidth(1.5)
    
#     plt.tight_layout()
#     plt.savefig('../result/entr_compl_plots/entropy_complexity_genre_means.png', dpi=300, bbox_inches='tight')
#     plt.show()


if __name__ == "__main__":
    df = pd.read_csv('./test_metrics_results.csv')
    plot_bar_chart(df, 'mdl_complexity', 'MDL Complexity')
    # plot_bar_chart(df, 'zip_compression_ratio', 'ZIP Compression')
    # plot_test_entropy_complexity(df)