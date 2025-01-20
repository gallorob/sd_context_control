import os
import PIL.Image
import numpy as np
import pandas as pd
from tqdm.auto import trange
from PIL import ImageDraw, ImageFont
import itertools
import matplotlib.pyplot as plt

runs_dir = './image_context_test_results/ddstyle'
save_dir = './image_context_test_results/user_study_imgs'

rng = np.random.default_rng(12345)

context_levels = ['no_context', 'colors_context', 'semantic_context', 'semantic_and_colors_context',
                  'semantic_and_image_context', 'caption_context', 'caption_and_colors_context',
                  'caption_and_image_context']
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

questionnaire_data = pd.DataFrame(columns=['Question Index', 'Run Number', 'Room Name', 'Entity Name', 'Contexts Order', 'Question Text'])

runs_dirs = [x for x in os.listdir(runs_dir) if x.startswith('sd_')]

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

n_imgs = 40
padding = 10
text_offset = (25, 10)
font = ImageFont.truetype("FreeMono.ttf", 100)
text_color = (255, 255, 255)

n_rows, n_cols = 1, 2
n_images_per_picture = 2  # len(context_levels)
assert n_rows * n_cols >= n_images_per_picture, f'You must have a grid that contains at minimum {n_images_per_picture} images.'

# idx = 0
all_combinations = list(itertools.product(runs_dirs, list(enemies.keys()), list(rooms.keys())))
rng.shuffle(all_combinations)

context_counter = {context: 0 for context in context_levels}

with trange(n_imgs, desc='Generating images for questionnaire...') as t:
	for i in t:
		# # choose random run, random entity, and random room without repetitions
		run_dir, entity_name, room_name = all_combinations[i]
		# prepare image
		room = PIL.Image.open(f'{runs_dir}/{run_dir}/{room_name.lower()}.png').convert('RGBA')
		q_img = PIL.Image.new(mode='RGB', size=(room.width * n_cols + padding * (n_cols - 1), room.height * n_rows + padding * (n_rows - 1)), color='#FFFFFF')
		draw = ImageDraw.Draw(q_img)
		
		context_idxs = np.arange(len(context_levels))
		rng.shuffle(context_idxs)
		context_idxs = context_idxs[:n_images_per_picture]
		contexts_order = ';'.join([context_levels[v] for v in context_idxs])
		
		for j, context_idx in enumerate(context_idxs):
			context_level = context_levels[context_idx]
			context_counter[context_level] += 1
			entity = PIL.Image.open(f'{runs_dir}/{run_dir}/{entity_name.lower()}_{room_name}_{context_level}.png').convert('RGBA')
			entity = entity.resize((int((room.height / entity.height) * entity.width), room.height), 0)
			room_with_entity = room.copy()
			room_with_entity.paste(entity, (room_with_entity.width // 2 - entity.width // 2, 0), entity)
			# copy over composited image
			col = j % n_cols
			row = (j - col) // n_cols
			q_img.paste(room_with_entity, (col * room.width + col * padding, row * room.height + row * padding))
			draw.text(xy=(col * room.width + col * padding + text_offset[0], row * room.height + row * padding + text_offset[1]), text=letters[j], fill=text_color, font=font)
		q_img.save(f'{save_dir}/{i}.png')
		questionnaire_data.loc[len(questionnaire_data)] = [i,
		                                                   run_dir,
		                                                   room_name,
		                                                   entity_name,
		                                                   contexts_order,
		                                                   f"{'Which enemy image fits its surroundings best?' if i < n_imgs / 2 else 'Which enemy image do you like more?'} These images depict a {entity_name} in {room_name}."]
		
questionnaire_data.to_csv(f'{save_dir}/questionnaire_data.csv')

plt.figure(figsize=(10, 6))
plt.bar(x=list(context_counter.keys()), height=[v / n_imgs for v in context_counter.values()], color='skyblue')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel('Context Types', fontsize=12)
plt.ylabel('Context Level Occurrence (%)', fontsize=12)
plt.title('Context Levels Occurrences')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{save_dir}/context_levels_stats.png', transparent=True)

