import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

plt.rcParams['font.family'] = 'Serif'

plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18

cblind_colors = ['#0072B2', '#D55E00', '#F0E422', '#009E73', '#CC79A7']
cblind_hatchings = ['/', '\\', '...', '-', 'x']

imgs_dir = './image_context_test_results/user_study_imgs'
save_dir = './image_context_test_results'
questionnaire_data = 'questionnaire_data.csv'
questionnaire_results = 'LLMAKER_Entities_Sprites_Evaluation_v3_Responses_Form_responses.csv'

raw_runs_dir = 'image_context_test_results/ddstyle'
raw_metrics_results = './image_context_test_results/ddstyle/results.csv'

data = pd.read_csv(os.path.join(imgs_dir, questionnaire_data))
results = pd.read_csv(os.path.join(save_dir, questionnaire_results))
raw_metrics = pd.read_csv(raw_metrics_results)

context_levels = ['no_context', 'colors_context', 'semantic_context', 'semantic_and_colors_context',
                  'semantic_and_image_context', 'caption_context', 'caption_and_colors_context',
                  'caption_and_image_context']

abbr_context_levels = ['None', 'Col', 'Sem', 'SemCol', 'SemImg', 'Cap', 'CapCol', 'CapImg']

context_to_abbr = {context_levels[i]: abbr_context_levels[i] for i in range(len(context_levels))}

metrics = ['Image Quality', 'Image Visibility', 'Image Consistency',
           'Image Complexity', 'Image Colorfulness']

extended_metrics = ['Image Quality', 'Image Visibility', 'Image Consistency',
                    'Image Complexity', 'Image Colorfulness',
                    'Average Image Diversity (within run)', 'Average Image Diversity (across runs)',
                    'Average Image Diversity (across rooms)']

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

dfq1 = pd.DataFrame(contexts_picks_q1)
dfq1.to_csv(os.path.join(save_dir, 'q1_picks.csv'))
dfq2 = pd.DataFrame(contexts_picks_q2)
dfq2.to_csv(os.path.join(save_dir, 'q2_picks.csv'))


def normalize_scores(scores: Dict[str, int],
                     counter: Dict[str, int]) -> Dict[str, float]:
	normalized_scores = {}
	for context in scores.keys():
		normalized_scores[context] = scores[context] / counter[context]
	return normalized_scores


#
#
contexts_score_q1_norm = normalize_scores(contexts_score_q1, contexts_counter_q1)
contexts_score_q2_norm = normalize_scores(contexts_score_q2, contexts_counter_q2)
#
x = np.arange(len(context_levels))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars_A = ax.bar(x - width / 2, contexts_score_q1_norm.values(), width, label='Q1', color=cblind_colors[0],
                edgecolor='black', hatch=cblind_hatchings[0], linewidth=1)
bars_B = ax.bar(x + width / 2, contexts_score_q2_norm.values(), width, label='Q2', color=cblind_colors[1],
                edgecolor='black', hatch=cblind_hatchings[1], linewidth=1)
ax.set_xlabel('Context Type')
ax.set_ylabel('Normalized Scores')
ax.set_xticks(x)
ax.set_xticklabels(abbr_context_levels, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(f'{save_dir}/qs_comparison.pdf', format='pdf', transparent=True)

ordered_q1_scores = {k: v for k, v in sorted(contexts_score_q1_norm.items(), key=lambda item: item[1], reverse=True)}
ordered_q2_scores = {k: contexts_score_q2_norm[k] for k in ordered_q1_scores.keys()}
#
x = np.arange(len(context_levels))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
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
plt.tight_layout()
plt.savefig(f'{save_dir}/qs_comparison_ordered.pdf', format='pdf', transparent=True)

contexts_q1_ranked = context_levels.copy()
contexts_q1_ranked.sort(key=lambda x: contexts_score_q1_norm[x])
contexts_q2_ranked = context_levels.copy()
contexts_q2_ranked.sort(key=lambda x: contexts_score_q2_norm[x])

context_frequency_q1 = {context: int(contexts_counter_q1[context] / len(results.index)) for context in context_levels}
context_frequency_q2 = {context: int(contexts_counter_q2[context] / len(results.index)) for context in context_levels}

with open(os.path.join(save_dir, 'questionnaire_analysis.log'), 'w') as f:
	f.write(f'Q1 context frequency: {context_frequency_q1}\n')
	f.write(f'Q2 context frequency: {context_frequency_q2}\n\n')
	
	f.write(f'Q1 normalized context scores: {contexts_score_q1_norm}\n')
	f.write(f'Q2 normalized context scores: {contexts_score_q2_norm}\n\n')
	
	f.write(f'Ranked contexts (Q1): {contexts_q1_ranked}\n')
	f.write(f'Ranked contexts (Q2): {contexts_q2_ranked}\n\n')

contexts_q1_order = [context_levels.index(x) for x in contexts_q1_ranked]
contexts_q2_order = [context_levels.index(x) for x in contexts_q2_ranked]

with open(os.path.join(save_dir, 'questionnaire_analysis.log'), 'a') as f:
	tau, kendall_pvalue = kendalltau(contexts_q1_order, contexts_q2_order)
	f.write(f'Kendall rankings correlation Q1-Q2: {tau=} (p={kendall_pvalue})\n')

metrics_contexts_scores = {k1: {k2: 0 for k2 in context_levels} for k1 in metrics}
experiments = [x for x in os.listdir(raw_runs_dir) if os.path.isdir(f'{raw_runs_dir}/{x}') and x.startswith('sd_')]

for i in range(col_offset, col_offset + nqs1):
	responses = results[results.columns[i]].tolist()
	a, b = data.loc[i - col_offset]['Contexts Order'].split(';')
	run_n = experiments.index(data.loc[i - col_offset]['Run Number'])
	room_name = data.loc[i - col_offset]['Room Name']
	entity_name = data.loc[i - col_offset]['Entity Name']
	metrics_a = raw_metrics[(raw_metrics['Run Number'] == run_n) &
	                        (raw_metrics['Room Name'] == room_name) &
	                        (raw_metrics['Entity Name'] == entity_name) &
	                        (raw_metrics['Context Level'] == a)].squeeze()
	metrics_b = raw_metrics[(raw_metrics['Run Number'] == run_n) &
	                        (raw_metrics['Room Name'] == room_name) &
	                        (raw_metrics['Entity Name'] == entity_name) &
	                        (raw_metrics['Context Level'] == b)].squeeze()
	for metric in metrics:
		m_a = metrics_a[metric]
		m_b = metrics_b[metric]
		if m_a > m_b:
			metrics_contexts_scores[metric][a] += 1
			metrics_contexts_scores[metric][b] -= 1
		elif m_b > m_a:
			metrics_contexts_scores[metric][b] += 1
			metrics_contexts_scores[metric][a] -= 1

for metric in metrics:
	metric_scores = metrics_contexts_scores[metric]
	x = np.arange(len(context_levels))
	width = 0.35
	fig, ax = plt.subplots(figsize=(10, 6))
	bars_A = ax.bar(x - width / 2, metric_scores.values(), width, color='skyblue')
	ax.set_xlabel('Context Type')
	ax.set_ylabel('Normalized Scores')
	ax.set_title(metric)
	ax.set_xticks(x)
	ax.set_xticklabels(context_levels, rotation=45, ha="right")
	ax.legend()
	plt.tight_layout()
	plt.savefig(f'{save_dir}/qs_{metric}_comparison.png', transparent=True)

metrics_contexts_rankings = {k1: context_levels.copy() for k1 in metrics}
for k in metrics_contexts_rankings:
	metrics_contexts_rankings[k].sort(key=lambda x: metrics_contexts_scores[k][x])

with open(os.path.join(save_dir, 'questionnaire_analysis.log'), 'a') as f:
	for qn, qdata in zip(['Q1', 'Q2'], [contexts_q1_order, contexts_q2_order]):
		for metric in metrics:
			metric_rankings = [context_levels.index(x) for x in metrics_contexts_rankings[metric]]
			
			tau, kendall_pvalue = kendalltau(metric_rankings, qdata)
			f.write(f'Kendall rankings correlation {qn}-{metric}: {tau=} (p={kendall_pvalue})\n')

rooms = {
	# Lost City of Atlantis
	"Crystal Grotto": "A cavern filled with luminescent crystals, casting an otherworldly glow, revealing hidden passages and secrets.",
	"Sunken Library": "An underwater chamber containing ancient scrolls and artifacts, guarded by spectral guardians.",
	"Forgotten Throne Room": "A grand hall adorned with coral and pearls, where the ghostly echoes of an ancient ruler still linger.",
	"Tidal Observatory": "A chamber housing a complex mechanism used by Atlantean scholars to observe celestial movements, now overrun by aquatic monsters.",
	"Submerged Arena": "An amphitheater where gladiatorial battles once took place, now inhabited by fierce aquatic beasts.",
	# Steampunk Sky Fortress
	"Cogwork Workshop": "A chamber filled with gears and steam-powered machinery, where mechanical creations are assembled and maintained.",
	"Airship Docking Bay": "A vast hangar housing airships of various sizes, bustling with activity as crews prepare for departure.",
	"Clocktower Observatory": "A towering structure equipped with telescopes and gears, offering a panoramic view of the surrounding sky.",
	"Steam Engine Room": "The heart of the fortress, where massive boilers and turbines generate power for the entire complex.",
	"Tesla Coil Chamber": "A laboratory crackling with electricity, where mad scientists conduct experiments with lightning and energy.",
	# Dark Faerie Woodlands
	"Enchanted Glade": "A serene clearing bathed in moonlight, inhabited by playful faeries and shimmering wisps.",
	"Shadowy Thicket": "A dense forest shrouded in darkness, where twisted trees and thorns conceal hidden dangers.",
	"Goblin Market": "A bustling marketplace where mischievous goblins barter and trade stolen treasures under the watchful eyes of their queen.",
	"Ethereal Nexus": "A mystical nexus where the boundaries between realms blur, granting access to otherworldly powers and creatures.",
	"Ancient Ruins": "Crumbling remnants of an ancient civilization, overgrown with moss and ivy, haunted by vengeful spirits.",
	# Post-Apocalyptic Cyberpunk Megacity
	"Neon Alley": "A bustling street lined with holographic advertisements and neon signs, where gangs and mercenaries vie for control.",
	"Cybernetic Bazaar": "A sprawling marketplace where illegal cybernetic enhancements and black-market technology are bought and sold.",
	"Data Haven": "A hidden enclave where hackers and information brokers gather to exchange valuable data and secrets.",
	"Corporate Skyscraper": "A towering edifice controlled by powerful corporations, filled with security checkpoints and automated defenses.",
	"Underground Sewer Network": "A maze of tunnels and conduits inhabited by scavengers, mutants, and rogue AI, hidden beneath the city streets.",
	# Ancient Egyptian Tomb Complex
	"Pharaoh's Chamber": "The burial chamber of a long-forgotten pharaoh, filled with treasure and guarded by ancient curses.",
	"Serpentine Labyrinth": "A maze of twisting corridors and deadly traps, designed to confound and ensnare intruders.",
	"Hieroglyphic Hallway": "A corridor adorned with intricate hieroglyphs and murals, depicting the deeds and rituals of the ancient Egyptians.",
	"Treasure Vault": "A chamber filled with golden artifacts and jeweled treasures, tempting plunderers with untold riches.",
	"Chamber of Anubis": "A sacred chamber dedicated to the jackal-headed god of death, where the spirits of the deceased are judged."
}

enemies = {
	# Lost City of Atlantis
	"Spectral Guardian": "Ethereal beings tasked with protecting the secrets of Atlantis, wielding energy beams and teleportation.",
	"Deep Sea Leviathan": "Gigantic sea creatures with armored scales and razor-sharp teeth, lurking in the dark depths.",
	"Coral Golem": "Constructs formed from enchanted coral, capable of regenerating and reshaping their bodies.",
	"Luminescent Jellyfish Swarm": "Bioluminescent creatures that stun their prey with electric shocks, attacking in large groups.",
	"Atlantean Warlock": "Masters of ancient magic, wielding power over water and summoning elemental creatures to their aid.",
	# Steampunk Sky Fortress
	"Clockwork Sentry": "Automated guardians equipped with rotating gears and steam-powered weaponry, relentless in their pursuit of intruders.",
	"Aetheric Drone": "Flying drones armed with tesla coils, capable of delivering powerful electric shocks from above.",
	"Steam-powered Golem": "Massive constructs fueled by steam, possessing immense strength and durability.",
	"Cogwork Spider": "Agile mechanical arachnids that scuttle across walls and ceilings, deploying traps and explosives.",
	"Mad Tinkerer": "Inventors driven mad by their experiments, armed with makeshift weapons and volatile gadgets.",
	# Dark Faerie Woodlands
	"Mischievous Imp": "Small, agile creatures known for their trickery and deception, wielding enchanted daggers and arcane spells.",
	"Shadow Wraith": "Malevolent spirits that lurk in the darkness, draining the life force of their victims with chilling touches.",
	"Thorned Treant": "Massive tree-like creatures adorned with razor-sharp thorns, capable of entangling and crushing their prey.",
	"Banshee Banshee": "Ghostly apparitions wailing mournful cries, capable of inflicting fear and despair with their haunting presence.",
	"Faerie Queen's Guard": "Elite warriors sworn to protect the faerie queen, armed with enchanted blades and shields, and capable of flight.",
	# Post-Apocalyptic Cyberpunk Megacity
	"Cyberpunk Mercenary": "Ruthless mercenaries augmented with cybernetic enhancements, armed with high-tech weaponry and tactical gear.",
	"Mutant Enforcer": "Monstrous mutants mutated by exposure to toxic waste, possessing immense strength and resilience.",
	"Corporate Security Drone": "Automated drones equipped with surveillance systems and lethal weaponry, programmed to eliminate intruders.",
	"Hacktivist Hacker": "Skilled hackers fighting against corporate oppression, capable of infiltrating systems and disabling security measures.",
	"Rogue AI Sentinel": "Advanced artificial intelligence programs gone rogue, controlling drones and security systems to protect their territory.",
	# Ancient Egyptian Tomb Complex
	"Undead Guardian": "Reanimated corpses of ancient warriors, bound by dark magic to protect the tombs of their pharaohs.",
	"Scarab Swarm": "Swarms of flesh-eating beetles that emerge from cracks in the walls, devouring anything in their path.",
	"Anubian Sentinel": "Dog-headed statues brought to life by ancient curses, armed with spears and shields.",
	"Mummy Priest": "Undead priests skilled in dark rituals and ancient curses, capable of draining the life force of their foes.",
	"Sphinx Guardian": "Majestic creatures with the body of a lion and the head of a human, posing riddles to those who seek passage through their domain."
}

if not os.path.exists(os.path.join(save_dir, 'full_metrics_contexts_scores.json')):
	full_metrics_contexts_scores = {k1: {k2: 0 for k2 in context_levels} for k1 in extended_metrics}
	for run_n, _ in enumerate(experiments):
		for room_name in rooms.keys():
			for entity_name in enemies.keys():
				for i, context_level_a in enumerate(context_levels):
					for context_level_b in context_levels[i:]:
						if context_level_a != context_level_b:
							metrics_a = raw_metrics[(raw_metrics['Run Number'] == run_n) &
							                        (raw_metrics['Room Name'] == room_name) &
							                        (raw_metrics['Entity Name'] == entity_name) &
							                        (raw_metrics['Context Level'] == context_level_a)].squeeze()
							metrics_b = raw_metrics[(raw_metrics['Run Number'] == run_n) &
							                        (raw_metrics['Room Name'] == room_name) &
							                        (raw_metrics['Entity Name'] == entity_name) &
							                        (raw_metrics['Context Level'] == context_level_b)].squeeze()
							for metric in extended_metrics:
								print(
									f'Comparing {run_n}-{room_name}-{entity_name}: {context_level_a} vs {context_level_b} on {metric}')
								m_a = metrics_a[metric]
								m_b = metrics_b[metric]
								if m_a > m_b:
									full_metrics_contexts_scores[metric][context_level_a] += 1
									full_metrics_contexts_scores[metric][context_level_b] -= 1
								elif m_b > m_a:
									full_metrics_contexts_scores[metric][context_level_b] += 1
									full_metrics_contexts_scores[metric][context_level_a] -= 1
	
	with open(os.path.join(save_dir, 'full_metrics_contexts_scores.json'), 'w') as f:
		json.dump(full_metrics_contexts_scores, f)
else:
	with open(os.path.join(save_dir, 'full_metrics_contexts_scores.json'), 'r') as f:
		full_metrics_contexts_scores = json.load(f)

tot_n = 6250 * 8
for metric in full_metrics_contexts_scores:
	for context_level in full_metrics_contexts_scores[metric]:
		full_metrics_contexts_scores[metric][context_level] /= tot_n

full_metrics_contexts_rankings = {k1: context_levels.copy() for k1 in extended_metrics}
for k in full_metrics_contexts_rankings:
	full_metrics_contexts_rankings[k].sort(key=lambda x: full_metrics_contexts_scores[k][x], reverse=True)

with open(os.path.join(save_dir, 'questionnaire_analysis.log'), 'a') as f:
	for metric_a in extended_metrics:
		metric_rankings_a = [context_levels.index(x) for x in full_metrics_contexts_rankings[metric_a]]
		f.write(f'\n{metric_a} ranking: {full_metrics_contexts_rankings[metric_a]}\n')
		for metric_b in extended_metrics:
			if metric_a != metric_b:
				metric_rankings_b = [context_levels.index(x) for x in full_metrics_contexts_rankings[metric_b]]
				
				tau, kendall_pvalue = kendalltau(metric_rankings_a, metric_rankings_b)
				f.write(f'Kendall rankings correlation {metric_a}-{metric_b}: {tau=} (p={kendall_pvalue})\n')

plotting_metrics = ['Image Quality', 'Image Consistency',
                    'Image Complexity', 'Average Image Diversity (across rooms)']
metrics_legend = ['Quality', 'Consistency', 'Complexity', 'Diversity']

x = np.arange(len(context_levels))
width = 1 / len(plotting_metrics)
fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(plotting_metrics):
	ax.bar(x - (i * width), full_metrics_contexts_scores[metric].values(), width, label=metrics_legend[i],
	       color=cblind_colors[i], edgecolor='black', hatch=cblind_hatchings[i], linewidth=1
	       )
ax.set_xlabel('Context Type')
ax.set_ylabel('Scores')
ax.set_xticks(x - (width * (len(plotting_metrics) / 2)))
ax.set_xticklabels(abbr_context_levels, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(f'{save_dir}/data_comparison.pdf', format='pdf', transparent=True)

ordered_contexts = full_metrics_contexts_rankings['Image Quality']
ordered_scores = {metric: [full_metrics_contexts_scores[metric][context] for context in ordered_contexts] for metric in
                  plotting_metrics}
ordered_context_labels = [abbr_context_levels[context_levels.index(context)] for context in ordered_contexts]

x = np.arange(len(ordered_contexts))
width = 1 / (len(plotting_metrics) + 1)
fig, ax = plt.subplots(figsize=(12, 8))
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
plt.tight_layout()
plt.savefig(f'{save_dir}/data_comparison_v2.pdf', format='pdf', transparent=True)
