import os
import warnings
from collections import Counter
from typing import List, Optional

import PIL.Image
import numpy as np
import pandas as pd
import torch as th
from lpips import LPIPS
from piq import brisque
from scipy.spatial import KDTree
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm.auto import trange
from webcolors import CSS3_HEX_TO_NAMES, HTML4_HEX_TO_NAMES, hex_to_rgb
from controlnet_aux import HEDdetector

metrics = ['Image Quality',
           # 'Image Visibility',
           'Image Consistency',
           'Image Complexity',
           # 'Image Colorfulness',
           # 'Average Image Diversity (within run)',
           # 'Average Image Diversity (across runs)',
           'Average Image Diversity (across rooms)'
           ]

results_data = pd.DataFrame(columns=['Run Number', 'Room Name', 'Entity Name', 'Context Level',
                                     *metrics
                                     ])

device = 'cuda' if th.cuda.is_available() else 'cpu'

if device == 'cpu':
	warnings.warn('CUDA not available; this will take much longer...')

lpips_eval = LPIPS(net='alex', lpips=True).to(device)
hed = HEDdetector.from_pretrained('lllyasviel/ControlNet', cache_dir='./models')

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

working_dir = './image_context_test_results/no_style'

context_levels = ['no_context', 'colors_context', 'semantic_context', 'semantic_and_colors_context',
                  'semantic_and_image_context', 'caption_context', 'caption_and_colors_context',
                  'caption_and_image_context']

max_n_colors = 32
max_colors_to_sd = max_n_colors // 4
color_names = 'html4'  # 'css3'

entity_image_size = (768, 512)

base_rng_seed = 1234
n_runs = 10


def compute_image_quality(image: PIL.Image) -> float:
	image_arr = th.from_numpy(np.expand_dims(np.array(image), 0)).movedim(3, 1).to(device) / 255.
	return brisque(x=image_arr[:, 0:3, :, :]).detach().cpu().item()


def compute_image_visibility(image: PIL.Image,
                             cropped_background: PIL.Image) -> float:
	background_copy = cropped_background.copy()
	background_copy.paste(image)
	
	grayscaled_background = np.asarray(cropped_background.convert('L'))
	grayscaled_composited_background = np.asarray(background_copy.convert('L'))
	
	return np.mean(np.sqrt(np.square(grayscaled_composited_background - grayscaled_background))) / np.count_nonzero(
		np.array(image)[:, :, 3])


def edge_complexity(image: PIL.Image):
	edges = hed(image).convert("L")
	# Count the number of white pixels in the edge map and normalize it by the total number of pixels
	num_edges = np.count_nonzero(edges) / float(edges.size[0] * edges.size[1])
	# Return the image complexity as a number between 0 and 1, where 1 means the image is highly complex
	return num_edges


def get_colorfulness(img: PIL.Image):
    l_rgbR, l_rgbG, l_rgbB, _ = img.split()
    l_rgbR = np.array(l_rgbR).astype(float) / 255.
    l_rgbG = np.array(l_rgbG).astype(float) / 255.
    l_rgbB = np.array(l_rgbB).astype(float) / 255.
    l_rg = l_rgbR - l_rgbG
    l_yb = 0.5 * l_rgbR + 0.5 * l_rgbG - l_rgbB
    rg_sd = np.std(l_rg)
    rg_mean = np.mean(l_rg)
    yb_sd = np.std(l_yb)
    yb_mean = np.mean(l_yb)
    rg_yb_sd = (rg_sd ** 2 + yb_sd ** 2) ** 0.5
    rg_yb_mean = (rg_mean ** 2 + yb_mean ** 2) ** 0.5
    colorful = rg_yb_sd + (rg_yb_mean * 0.3)
    return colorful


def convert_rgb_to_names(rgb_tuple):
	names = []
	rgb_values = []
	ref_colors_dict = CSS3_HEX_TO_NAMES if color_names == 'css3' else HTML4_HEX_TO_NAMES
	for color_hex, color_name in ref_colors_dict.items():
		names.append(color_name)
		rgb_values.append(hex_to_rgb(color_hex))
	
	kdt_db = KDTree(rgb_values)
	distance, index = kdt_db.query(rgb_tuple)
	return names[index]


def compute_image_consistency(image: PIL.Image,
                              background: PIL.Image) -> float:
	def get_colors(quantized_image: PIL.Image) -> List[str]:
		all_colors = quantized_image.getcolors()
		all_colors_names = [convert_rgb_to_names(x[1]) for x in all_colors]
		colors_counts = Counter(all_colors_names)
		return list(sorted(colors_counts.keys(), key=lambda x: colors_counts[1], reverse=True))
	
	quantized_background = background.quantize(max_n_colors).convert('RGB')
	quantized_image = image.quantize(max_n_colors).convert('RGB')
	
	set_colors_background = set(get_colors(quantized_background))
	set_colors_entity = set(get_colors(quantized_image))
	
	return len(set_colors_background.intersection(set_colors_entity)) / len(
		set_colors_background.union(set_colors_entity))


def compute_image_diversity(image: PIL.Image,
                            other_image: PIL.Image.Image,
                            # cropped_background: PIL.Image.Image,
                            # other_cropped_background: Optional[PIL.Image.Image] = None
                            ) -> float:
	# image_with_background = cropped_background.copy()
	# image_with_background.paste(image, mask=image)
	# image_with_background = image_with_background.convert('RGB')
	
	# other_image_with_background = other_cropped_background if other_cropped_background is not None else cropped_background.copy()
	# other_image_with_background.paste(other_image, mask=other_image)
	# other_image_with_background = other_image_with_background.convert('RGB')
	
	image = image.convert('RGB')
	other_image = other_image.convert('RGB')

	# image_with_background_tensor = th.from_numpy(np.array(image_with_background)).to(device)
	# image_with_background_tensor = th.reshape(image_with_background_tensor, (1, 3, image.size[1], image.size[0]))
	# other_image_with_background_tensor = th.from_numpy(np.array(other_image_with_background)).to(device)
	# other_image_with_background_tensor = th.reshape(other_image_with_background_tensor,
	#                                                 (1, 3, image.size[1], image.size[0]))
	
	image_tensor = th.from_numpy(np.array(image)).to(device)
	image_tensor = th.reshape(image_tensor, (1, 3, image.size[1], image.size[0]))
	other_image_tensor = th.from_numpy(np.array(other_image)).to(device)
	other_image_tensor = th.reshape(other_image_tensor, (1, 3, image.size[1], image.size[0]))
	
	return lpips_eval.forward(image_tensor, other_image_tensor).detach().cpu().numpy()[0]


t = trange(n_runs * len(rooms) * len(enemies), desc='Compiling metrics')

experiments = [x for x in os.listdir(working_dir) if os.path.isdir(f'{working_dir}/{x}') and x.startswith('sd_')]
for k, exp_name in enumerate(experiments):
	for i, (room_name, room_description) in enumerate(rooms.items()):
		for j, (entity_name, entity_description) in enumerate(enemies.items()):
			t.n += 1
			t.set_postfix_str(f'{exp_name} - {room_name}: {entity_name}')
			room_image_cropped = PIL.Image.open(f'{working_dir}/{exp_name}/{room_name}_cropped.png').convert('RGBA')
			room_image_cropped = room_image_cropped.resize((entity_image_size[1], entity_image_size[0]))
			for context_level in context_levels:
				entity_image = PIL.Image.open(
					f'{working_dir}/{exp_name}/{entity_name.lower()}_{room_name}_{context_level}.png').convert('RGBA')
				image_quality = compute_image_quality(entity_image)
				# image_visibility = compute_image_visibility(entity_image, room_image_cropped)
				image_consistency = compute_image_consistency(entity_image, room_image_cropped)
				image_complexity = edge_complexity(entity_image)
				# image_colorfulness = get_colorfulness(entity_image)
				# image_diversities_within_runs = []
				# for other_context_level in context_levels:
				# 	if context_level != other_context_level:
				# 		other_image = PIL.Image.open(
				# 			f'{working_dir}/{exp_name}/{entity_name.lower()}_{room_name}_{other_context_level}.png').convert('RGBA')
				# 		image_diversities_within_runs.append(
				# 			compute_image_diversity(entity_image, other_image, room_image_cropped))
				# image_diversities_across_runs = []
				# for other_exp_name in experiments:
				# 	if exp_name != other_exp_name:
				# 		other_image = PIL.Image.open(
				# 			f'{working_dir}/{other_exp_name}/{entity_name.lower()}_{room_name}_{context_level}.png').convert('RGBA')
				# 		image_diversities_across_runs.append(
				# 			compute_image_diversity(entity_image, other_image, room_image_cropped))
				image_diversities_across_rooms = []
				for other_room_name, _ in rooms.items():
					if room_name != other_room_name:
						# other_room_image_cropped = PIL.Image.open(
						# 	f'{working_dir}/{exp_name}/{other_room_name}_cropped.png')
						# other_room_image_cropped = other_room_image_cropped.resize(entity_image_size)
						other_image = PIL.Image.open(
							f'{working_dir}/{exp_name}/{entity_name.lower()}_{other_room_name}_{context_level}.png').convert('RGBA')
						image_diversities_across_rooms.append(
							compute_image_diversity(entity_image, other_image,
							                        # room_image_cropped, other_room_image_cropped
							                        ))
				results_data.loc[len(results_data)] = [k, room_name, entity_name, context_level,
				                                       image_quality,
				                                       # image_visibility,
				                                       image_consistency,
				                                       image_complexity,
				                                       # image_colorfulness,
				                                       # np.mean(image_diversities_within_runs),
				                                       # np.mean(image_diversities_across_runs),
				                                       np.mean(image_diversities_across_rooms)
				                                       ]

results_data.to_csv(f'{working_dir}/results.csv')

t = trange(len(metrics), desc='Processing results...')

processed_data = pd.DataFrame(
	columns=['Metric', 'Context Level', 'Mean', 'Standard Deviation', 'Count', '95% Confidence Interval'])

for i, metric in enumerate(metrics):
	t.n += 1
	t.set_postfix_str(metric)
	# Calculate mean and confidence intervals
	grouped = results_data.groupby(['Context Level']).agg({metric: ['mean', 'std', 'count']})
	grouped.columns = ['mean', 'std', 'count']
	grouped['ci'] = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
	
	for j, context_level in enumerate(context_levels):
		processed_data.loc[len(processed_data)] = [metric, context_level, grouped['mean'].values[j],
		                                           grouped['std'].values[j], grouped['count'].values[j],
		                                           grouped['ci'].values[j]]

# Plotting
# fig, ax = plt.subplots(figsize=(12, 8))
#
# sns.barplot(x=grouped.index, y='mean', data=grouped, ax=ax)
# ax.errorbar(x=grouped.index, y=grouped['mean'], yerr=grouped['ci'], fmt='none', capsize=5, capthick=2,
#             color='black')
#
# ax.set_ylabel(metric)
# ax.set_xlabel('Context Level')
# ax.set_title(f'{metric} across Context Levels')
#
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(f'{working_dir}/{metric}.png', transparent=True)

processed_data.to_csv(f'{working_dir}/processed_results.csv')

stats_df = pd.DataFrame(
	columns=['Metric', 'Group0', 'Group1', 'Statistic', 'p-value', 'Corrected p-value', 'Significant?'])

for context_level in context_levels:
	for metric in metrics:
		statistics, pvals, group1s = [], [], []
		for other_context_level in context_levels:
			if context_level != other_context_level:
				group0 = results_data.loc[(results_data['Context Level'] == context_level)][metric]
				group1 = results_data.loc[(results_data['Context Level'] == other_context_level)][metric]
				res = wilcoxon(group0, group1, zero_method='wilcox')
				statistics.append(res.statistic)
				pvals.append(res.pvalue)
				group1s.append(other_context_level)
		rejects, pvals_corrected, _, _ = multipletests(pvals=pvals, method='bonferroni')
		for group1, static, pval, pval_corrected, reject in zip(group1s, statistics, pvals, pvals_corrected, rejects):
			new_row = pd.DataFrame([{
				'Metric': metric, 'Group0': context_level, 'Group1': group1,
				'Statistic': static, 'p-value': pval, 'Corrected p-value': pval_corrected,
				'Significant?': reject
			}])
			stats_df = pd.concat([stats_df, new_row])

stats_df.to_csv(f'{working_dir}/stat_analysis.csv')

summary_df = pd.DataFrame(columns=['Metric', *context_levels])

for metric in metrics:
	ns = []
	for context_level in context_levels:
		n = 0
		v0 = \
			processed_data.loc[
				(processed_data['Metric'] == metric) & (processed_data['Context Level'] == context_level)][
				'Mean'].item()
		# print(f'{context_level=} {v0=}')
		for other_context_level in context_levels:
			if context_level != other_context_level:
				v1 = \
					processed_data.loc[(processed_data['Metric'] == metric) & (
							processed_data['Context Level'] == other_context_level)][
						'Mean'].item()
				s = stats_df.loc[(stats_df['Metric'] == metric) & (stats_df['Group0'] == context_level) & (
						stats_df['Group1'] == other_context_level)]['Significant?'].item()
				if v0 > v1 and s:
					n += 1
		ns.append(n)
	new_row = pd.DataFrame([{
		'Metric': metric,
		**{x: ns[i] for i, x in enumerate(context_levels)}
	}])
	summary_df = pd.concat([summary_df, new_row])

summary_df.to_csv(f'{working_dir}/summary_table.csv')

print(summary_df.to_latex())
