import json
import os
from typing import Dict

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageDraw, ImageFont
from matplotlib import font_manager

plt.rcParams['font.family'] = 'Serif'

plt.rcParams['font.size'] = 32
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 28

cblind_colors = ['#0072B2', '#D55E00', '#F0E422', '#009E73', '#CC79A7']
cblind_hatchings = ['/', '\\', '...', '-', 'x']

abbr_context_levels = ['NONE', 'COL', 'SEM', 'SEMCOL', 'SEMIMG', 'CAP', 'CAPCOL', 'CAPIMG']
context_levels = ['no_context', 'colors_context', 'semantic_context', 'semantic_and_colors_context',
                  'semantic_and_image_context', 'caption_context', 'caption_and_colors_context',
                  'caption_and_image_context']

context_to_abbr = {context_levels[i]: abbr_context_levels[i] for i in range(len(context_levels))}

metrics = ['Image Quality', 'Image Visibility', 'Image Consistency',
           'Image Complexity', 'Image Colorfulness']

extended_metrics = ['Image Quality', 'Image Visibility', 'Image Consistency',
                    'Image Complexity', 'Image Colorfulness',
                    'Average Image Diversity (within run)', 'Average Image Diversity (across runs)',
                    'Average Image Diversity (across rooms)']

# Generate 3x3 grid of entry with best quality, ordered by context
raw_metrics_results = './image_context_test_results/ddstyle/results.csv'
raw_metrics = pd.read_csv(raw_metrics_results)
metric = 'Image Quality'

best_quality_entry = raw_metrics.iloc[[np.argmax(raw_metrics[metric])]]
entry = raw_metrics[(raw_metrics['Run Number'] == best_quality_entry['Run Number'].item()) &
                    (raw_metrics['Room Name'] == best_quality_entry['Room Name'].item()) &
                    (raw_metrics['Entity Name'] == best_quality_entry['Entity Name'].item())]
entry_qualities = entry[metric]
context_levels = entry['Context Level']
ordered_contexts = context_levels.iloc[entry_qualities.argsort()[::-1]]
room_name = best_quality_entry['Room Name'].item()
run_n = best_quality_entry['Run Number'].item()
entity_name = best_quality_entry['Entity Name'].item()

runs_dir = './image_context_test_results/ddstyle'
runs_dirs = [x for x in os.listdir(runs_dir) if x.startswith('sd_')]
run_dir = runs_dirs[run_n]
n_cols = 3
n_rows = 3
padding = 10
text_offset = (25, 10)
font_name = font_manager.FontProperties(family='serif')
font_file = font_manager.findfont(font_name)
font = ImageFont.truetype(font_file, 60)
text_color = (255, 255, 255)
room = PIL.Image.open(f'{runs_dir}/{run_dir}/{room_name.lower()}.png').convert('RGBA')
q_img = PIL.Image.new(mode='RGB', size=(
room.width * n_cols + padding * (n_cols - 1), room.height * n_rows + padding * (n_rows - 1)), color='#FFFFFF00')
draw = ImageDraw.Draw(q_img)

for context_idx, context_level in enumerate(ordered_contexts):
	entity = PIL.Image.open(f'{runs_dir}/{run_dir}/{entity_name.lower()}_{room_name}_{context_level}.png').convert(
		'RGBA')
	entity = entity.resize((int((room.height / entity.height) * entity.width), room.height), 0)
	room_with_entity = room.copy()
	room_with_entity.paste(entity, (room_with_entity.width // 2 - entity.width // 2, 0), entity)
	# copy over composited image
	col = context_idx % n_cols
	row = (context_idx - col) // n_cols
	q_img.paste(room_with_entity, (col * room.width + col * padding, row * room.height + row * padding))
	text = f'#{context_idx + 1}: {abbr_context_levels[context_levels.index(context_level)]}'
	draw.text(
		xy=(col * room.width + col * padding + text_offset[0], row * room.height + row * padding + text_offset[1]),
		text=text, fill=text_color, font=font)
q_img.save(f'./image_context_test_results/example_ranking_{metric}_3x3.pdf', format='pdf', transparent=True)

# Generate the automated metrics plot
with open(os.path.join('backups/full_metrics_contexts_scores.json'), 'r') as f:
	full_metrics_contexts_scores = json.load(f)

tot_n = 6250 * 8
for metric in full_metrics_contexts_scores:
	for context_level in full_metrics_contexts_scores[metric]:
		full_metrics_contexts_scores[metric][context_level] /= tot_n

full_metrics_contexts_rankings = {k1: context_levels.copy() for k1 in extended_metrics}
for k in full_metrics_contexts_rankings:
	full_metrics_contexts_rankings[k].sort(key=lambda x: full_metrics_contexts_scores[k][x], reverse=True)

plotting_metrics = ['Image Quality', 'Image Consistency',
                    'Image Complexity', 'Average Image Diversity (across rooms)']
metrics_legend = ['Quality', 'Consistency', 'Complexity', 'Diversity']

ordered_contexts = full_metrics_contexts_rankings['Image Quality']
ordered_scores = {metric: [full_metrics_contexts_scores[metric][context] for context in ordered_contexts] for metric in
                  plotting_metrics}
ordered_context_labels = [abbr_context_levels[context_levels.index(context)] for context in ordered_contexts]

x = np.arange(len(ordered_contexts))
width = 1 / (len(plotting_metrics) + 1)
fig, ax = plt.subplots(figsize=(20, 8))
for i, metric in enumerate(plotting_metrics):
	ax.bar(x + (i * width),
	       ordered_scores[metric],
	       width,
	       label=metrics_legend[i],
	       color=cblind_colors[i], edgecolor='black', hatch=cblind_hatchings[i], linewidth=1
	       )
ax.set_xlabel('Context Type')
ax.set_ylabel('Scores')
ax.set_xticks(x + (width * (len(plotting_metrics) / 2)) - (width / 2))
ax.set_xticklabels(ordered_context_labels, rotation=45, ha="right")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axhline(0, 0, 1, color='black', linewidth=1)
plt.axvline(0.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(1.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(2.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(3.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(4.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(5.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(6.8, -1, 1, color='black', linewidth=1, linestyle='--')
plt.tight_layout()
plt.savefig(f'image_context_test_results/data_comparison_v2.pdf', format='pdf', transparent=True)

# Generate questionnaire results plot
contexts_score_q1 = {k: 0 for k in context_levels}
contexts_score_q2 = {k: 0 for k in context_levels}
#
contexts_counter_q1 = {context: 0 for context in context_levels}
contexts_counter_q2 = {context: 0 for context in context_levels}
#
nqs1 = 20
nqs2 = 20
col_offset = 2
#
contexts_picks_q1 = {context: {'Picked': 0, 'Not Picked': 0, 'Both': 0, 'Neither': 0} for context in context_levels}
contexts_picks_q2 = {context: {'Picked': 0, 'Not Picked': 0, 'Both': 0, 'Neither': 0} for context in context_levels}


#
def process_responses(ref_data: pd.DataFrame,
                      qas: pd.DataFrame,
                      start: int,
                      n_qs: int,
                      counter: Dict[str, int],
                      picks: Dict[str, Dict[str, int]],
                      score: Dict[str, int]) -> None:
	for i in range(start, start + n_qs):
		responses = qas[qas.columns[i]].tolist()
		a, b = ref_data.loc[i - col_offset]['Contexts Order'].split(';')
		for response in responses:
			counter[a] += 1
			counter[b] += 1
			if response == 'Both':
				score[a] += 1
				score[b] += 1
				picks[a]['Both'] += 1
				picks[b]['Both'] += 1
			elif response == 'A':
				score[a] += 1
				score[b] += -1
				picks[a]['Picked'] += 1
				picks[b]['Not Picked'] += 1
			elif response == 'B':
				score[a] += -1
				score[b] += 1
				picks[a]['Not Picked'] += 1
				picks[b]['Picked'] += 1
			elif response == 'Neither':
				score[a] += -1
				score[b] += -1
				
				picks[a]['Neither'] += 1
				picks[b]['Neither'] += 1
			else:
				raise ValueError(f'Unrecognized {response=}')


imgs_dir = './image_context_test_results/user_study_imgs'
save_dir = './image_context_test_results'
questionnaire_data = 'questionnaire_data.csv'
questionnaire_results = 'LLMAKER_Entities_Sprites_Evaluation_v3_Responses_Form_responses.csv'
data = pd.read_csv(os.path.join(imgs_dir, questionnaire_data))
results = pd.read_csv(os.path.join(save_dir, questionnaire_results))

process_responses(ref_data=data,
                  qas=results,
                  start=col_offset,
                  n_qs=nqs1,
                  counter=contexts_counter_q1,
                  picks=contexts_picks_q1,
                  score=contexts_score_q1)

process_responses(ref_data=data,
                  qas=results,
                  start=col_offset + nqs1,
                  n_qs=nqs2,
                  counter=contexts_counter_q2,
                  picks=contexts_picks_q2,
                  score=contexts_score_q2)


def normalize_scores(scores: Dict[str, int],
                     counter: Dict[str, int]) -> Dict[str, float]:
	normalized_scores = {}
	for context in scores.keys():
		normalized_scores[context] = scores[context] / counter[context]
	return normalized_scores


contexts_score_q1_norm = normalize_scores(contexts_score_q1, contexts_counter_q1)
contexts_score_q2_norm = normalize_scores(contexts_score_q2, contexts_counter_q2)

ordered_q1_scores = {k: v for k, v in sorted(contexts_score_q1_norm.items(), key=lambda item: item[1], reverse=True)}
ordered_q2_scores = {k: contexts_score_q2_norm[k] for k in ordered_q1_scores.keys()}
#
x = np.arange(len(context_levels))
width = 0.35
fig, ax = plt.subplots(figsize=(20, 8))
bars_A = ax.bar(x - width / 2, ordered_q1_scores.values(), width, label='Q1', color=cblind_colors[0], edgecolor='black',
                hatch=cblind_hatchings[0], linewidth=1)
bars_B = ax.bar(x + width / 2, ordered_q2_scores.values(), width, label='Q2', color=cblind_colors[1], edgecolor='black',
                hatch=cblind_hatchings[1], linewidth=1)
ax.set_xlabel('Context Type')
ax.set_ylabel('Normalized Scores')
ax.set_xticks(x)
ax.set_xticklabels([context_to_abbr[k] for k in ordered_q1_scores.keys()], rotation=45, ha="right")
ax.legend()
plt.axhline(0, 0, 1, color='black', linewidth=1)
plt.axvline(0.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(1.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(2.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(3.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(4.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(5.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.axvline(6.5, -1, 1, color='black', linewidth=1, linestyle='--')
plt.tight_layout()
# plt.show()
plt.savefig(f'image_context_test_results/qs_comparison_ordered.pdf', format='pdf', transparent=True)
