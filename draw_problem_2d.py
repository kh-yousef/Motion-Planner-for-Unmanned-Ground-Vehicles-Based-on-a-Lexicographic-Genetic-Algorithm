import matplotlib.pyplot as plt

def draw_problem_2d(op, gas, paths_xy, final_fitness_vector=None, fitness_history=None):
    # ax = plt.gca()
    # Ensure a figure exists and set layout engine to prevent clipping
    fig = plt.gcf()
    ax = plt.gca()
    
    ax.set_facecolor('#FFFFFF') 
    fig.set_facecolor('white') 
    
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    legend_handles = []
    legend_labels = []

    robot_radius = op.get('robot_radius', 0)
    robot_color = '#1f77b4' 
    robot_alpha = 0.3

    if 'environment_bounds' in op and op['environment_bounds'] is not None:
        bounds = op['environment_bounds']
        rect = plt.Rectangle((bounds[0], bounds[2]), bounds[1] - bounds[0], bounds[3] - bounds[2],
                             edgecolor=(0.301, 0.745, 0.933, 0.8), linestyle='--', linewidth=2, fill=False, label='Environment')
        ax.add_patch(rect)
        legend_handles.append(rect)
        legend_labels.append('Boudaries')

    h_home, = plt.plot(op['home_base'][0], op['home_base'][1], 's', markersize=15, markerfacecolor='white', markeredgecolor='black', label='Home Base')
    legend_handles.append(h_home)
    legend_labels.append('Home Base')

    if 'obstacles' in op and op['obstacles'] is not None and len(op['obstacles']) > 0:
        for i, obs in enumerate(op['obstacles']):
            obs_circle = plt.Circle((obs[0], obs[1]), obs[2], color='#808080', alpha=0.9, label='Obstacles' if i == 0 else "")
            ax.add_patch(obs_circle)
        if len(op['obstacles']) > 0:
            obstacle_proxy = plt.Rectangle((0, 0), 1, 1, fc='#808080', alpha=0.9)
            legend_handles.append(obstacle_proxy)
            legend_labels.append('Obstacles')

    if 'targets' in op and op['targets'] is not None and len(op['targets']) > 0:
        for i, target in enumerate(op['targets']):
            h_target, = plt.plot(target[0], target[1], '*', markersize=25, markerfacecolor='red', markeredgecolor='white', label='Target')
            orientation_segment = op['end_points'][i]
            h_orient, = plt.plot(orientation_segment[:, 0], orientation_segment[:, 1], '--', color='red', linewidth=1.5, label='Target Orientation')
            if i == 0:
                legend_handles.extend([h_target, h_orient])
                legend_labels.extend(['Target', 'Target Orientation'])
                
    if paths_xy is not None and len(paths_xy) > 0:
        path = paths_xy[0] 
        h_path, = plt.plot(path[:, 0], path[:, 1], 'o-', color='#1f77b4', linewidth=2, markersize=5, markerfacecolor='#1f77b4', label='Robot Path')
        legend_handles.append(h_path)
        legend_labels.append('Robot Path')
        
        for i, node_pos in enumerate(path):
            body_circle = plt.Circle(node_pos, robot_radius, color=robot_color, alpha=robot_alpha, label='Robot Body' if i == 0 else "")
            ax.add_patch(body_circle)

        if len(path) > 0:
            body_proxy = plt.Rectangle((0, 0), 1, 1, fc=robot_color, alpha=robot_alpha)
            legend_handles.append(body_proxy)
            legend_labels.append('Robot Body')

               # if final_fitness_vector is not None:
    #     title_str_line1 = (f"Optimal Path | IK Fitness: {final_fitness_vector[gas['fitIdx']['ikFitness']]:.2f}, "
    #                        f"Length: {final_fitness_vector[gas['fitIdx']['pathLength']]:.1f}")
    #     title_str_line2 = (f"Segments: {int(round(final_fitness_vector[gas['fitIdx']['nodeCount']]))}, "
    #                        f"Undulation: {final_fitness_vector[gas['fitIdx']['undulation']]:.2f}, "
    #                        f"Penalty: {final_fitness_vector[gas['fitIdx']['penalty']]:.2f}")
    #     # Added pad=15 to force vertical clearance between the top axis spine and multiline text
    #     plt.title(f"{title_str_line1}\n{title_str_line2}", color='black', pad=15, fontsize=15)
    # else:
    #     plt.title('Decoded Path Visualization', color='black', pad=15, fontsize=15)
        
    plt.xlabel('X', color='black', fontsize=30)
    plt.ylabel('Y', color='black', fontsize=30)
    ax.tick_params(axis='x', colors='black', labelsize=30)
    ax.tick_params(axis='y', colors='black', labelsize=30)
    
    if 'environment_bounds' in op and op['environment_bounds'] is not None:
        bounds = op['environment_bounds']
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[2], bounds[3])
    
    plt.axis('equal')

    # legend = ax.legend(handles=legend_handles, labels=legend_labels, facecolor='white', edgecolor='black', framealpha=0.8, fontsize=15)
    # for text in legend.get_texts():
    #     text.set_color('black')
    
    # Enforce layout constraint to prevent the title and labels from being clipped outside the figure boundaries
    # plt.tight_layout()

