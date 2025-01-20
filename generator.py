import os
import warnings
from collections import Counter
from datetime import datetime
from typing import List

import PIL.Image
import numpy as np
import torch as th
from PIL import ImageFilter
from PIL.ImageOps import invert
from compel import Compel
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline, \
	StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from rembg import remove as remove_background
from scipy.spatial import KDTree
from tqdm.auto import trange
from transformers import BlipForConditionalGeneration, BlipProcessor
from webcolors import CSS3_HEX_TO_NAMES, HTML4_HEX_TO_NAMES, hex_to_rgb

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

max_n_colors = 32
max_colors_to_sd = max_n_colors // 4
color_names = 'html4'  # 'css3'

base_rng_seed = 1234
n_runs = 10

room_control_image = 'controlnet_masks/room_mask_v2.png'

# room_prompt = "ddstyle, a (side view)+++ screenshot of a room set in {room_name}, {room_description}, best quality"
room_prompt = "a (side view)+++ screenshot of a room set in {room_name}, {room_description}, best quality"
room_negative_prompt = "top-down, (logo)++, (creature)++, (people)++, (animal)++, (face)++, ugly, badly drawn, worst quality, frame, glare, solar flare, text, monochromatic, duplication, repetition"

entity_image_size = (768, 512)
entity_scale = 0.35

controlnet_guidance_scale = 5
sd_guidance_scale = 5
entity_inference_steps = 30
room_inference_steps = 30

device = 'cuda' if th.cuda.is_available() else 'cpu'

if device == 'cpu':
	warnings.warn('CUDA not available; this will take much longer...')

# vae = AutoencoderKL.from_single_file('models/vaeFtMse840000EmaPruned_vae.safetensors', torch_dtype=th.float16).to(device)

controlnet_mlsd = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_mlsd', torch_dtype=th.float16,
                                                  safety_checker=None, cache_dir='./models')
# sd_room = StableDiffusionControlNetPipeline.from_single_file('models/aZovyaRPGArtistTools_v4.safetensors',
sd_room = StableDiffusionControlNetPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                            cache_dir='./models',
                                                            safety_checker=None,
                                                            controlnet=controlnet_mlsd, torch_dtype=th.float16).to(
	device)
sd_room.safety_checker = None
sd_room.scheduler = DPMSolverMultistepScheduler.from_config(sd_room.scheduler.config, use_karras=True,
                                                            algorithm_type='sde-dpmsolver++')
sd_room.set_progress_bar_config(disable=True)
# sd_room.vae = vae
# sd_room.load_lora_weights('./models', weight_name='DarkestDungeonV2.safetensors')
compel_room = Compel(tokenizer=sd_room.tokenizer, text_encoder=sd_room.text_encoder, truncate_long_prompts=False)

# sd_entity = StableDiffusionPipeline.from_single_file('models/aZovyaRPGArtistTools_v4.safetensors',
sd_entity = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                    cache_dir='./models',
                                                    torch_dtype=th.float16, safety_checker=None).to(
	device)
sd_entity.scheduler = DPMSolverMultistepScheduler.from_config(sd_entity.scheduler.config, use_karras=True,
                                                              algorithm_type='sde-dpmsolver++')
sd_entity.safety_checker = None
sd_entity.set_progress_bar_config(disable=True)
# sd_entity.vae = vae
# sd_entity.load_lora_weights('./models', weight_name='NecroSketcherAlpha.safetensors')
compel_entity = Compel(tokenizer=sd_entity.tokenizer, text_encoder=sd_entity.text_encoder, truncate_long_prompts=False)

controlnet_softedge = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge',
                                                      torch_dtype=th.float16,
                                                      use_safetensors=True,
                                                      safety_checker=None, cache_dir='./models')
# sd_inpaint_entity = StableDiffusionControlNetInpaintPipeline.from_single_file(
# 	'models/aZovyaRPGArtistTools_v4.safetensors',
sd_inpaint_entity = StableDiffusionControlNetInpaintPipeline.from_pretrained(
	'runwayml/stable-diffusion-v1-5',
	cache_dir='./models',
	controlnet=controlnet_softedge,
	torch_dtype=th.float16,
	num_in_channels=4,
	use_safetensors=True,
	safety_checker=None).to(device)
sd_inpaint_entity.safety_checker = None
sd_inpaint_entity.scheduler = DPMSolverMultistepScheduler.from_config(sd_inpaint_entity.scheduler.config,
                                                                      use_karras=True, algorithm_type='sde-dpmsolver++')
sd_inpaint_entity.set_progress_bar_config(disable=True)
# sd_entity.vae = vae
# sd_inpaint_entity.load_lora_weights('./models', weight_name='NecroSketcherAlpha.safetensors')
compel_inpaint_entity = Compel(tokenizer=sd_inpaint_entity.tokenizer, text_encoder=sd_inpaint_entity.text_encoder,
                               truncate_long_prompts=False)


def clear_strings_for_prompt(strings: List[str]):
	return [s.lower().replace('.', '').replace('.', '').strip() for s in strings]


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


def prepare_cropped_image(room_image):
	w, h = room_image.width, room_image.height
	r = h / entity_image_size[0]
	
	scaled_entity_height = h
	scaled_entity_width = r * entity_image_size[1]
	x_offset = w / 2 - scaled_entity_width / 2
	
	# get the cropped background from room_image
	ref_image = room_image.copy()
	ref_image = ref_image.crop((x_offset,
	                            0,
	                            x_offset + scaled_entity_width,
	                            scaled_entity_height))
	ref_image.save(f'./image_context_test_results/{exp_name}/{room_name}_cropped.png')
	ref_image = ref_image.resize((entity_image_size[1],
	                              entity_image_size[0]))
	return ref_image


def generate_room(room_name, room_description):
	control_image = invert(load_image(room_control_image))
	room_name, room_description = clear_strings_for_prompt([room_name, room_description])
	formatted_prompt = room_prompt.format(room_name=room_name, room_description=room_description)
	with open(log_file, 'a') as f:
		f.write(f'ROOM: {room_name} {room_description} -> {formatted_prompt}\n')
	conditioning = compel_room.build_conditioning_tensor(formatted_prompt)
	negative_conditioning = compel_room.build_conditioning_tensor(room_negative_prompt)
	[conditioning, negative_conditioning] = compel_room.pad_conditioning_tensors_to_same_length(
		[conditioning, negative_conditioning])
	room_image = sd_room(image=control_image,
	                     prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
	                     num_inference_steps=room_inference_steps,
	                     guidance_scale=controlnet_guidance_scale,
	                     generator=th.Generator(device=device).manual_seed(base_rng_seed)).images[0]
	room_image.save(f'./image_context_test_results/{exp_name}/{room_name}.png')
	return room_image


def generate_entity(entity_name, entity_description,
                    context_level,
                    room_name=None, room_description=None,
                    colors_to_sd=None,
                    entity_context_image=None, cropped_room_image=None):
	# prompts = {
	# 	'no_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, (flat empty background)+++, masterpiece++, highly detailed+",
	# 	'colors_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, prevalent colors are {prevalent_colors}, (flat empty background)+++, masterpiece++, highly detailed+",
	# 	'semantic_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, set in {place_name}: {place_description}, (flat empty background)+++, masterpiece++, highly detailed+",
	# 	'semantic_and_colors_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, set in {place_name}: {place_description}, prevalent colors are {prevalent_colors}, (flat empty background)+++, masterpiece++, highly detailed+",
	# 	'semantic_and_image_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, set in {place_name}: {place_description}, masterpiece++, highly detailed+",
	# 	'caption_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, set in {place_description}, (flat empty background)+++, masterpiece++, highly detailed+",
	# 	'caption_and_colors_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, set in {place_description}, prevalent colors are {prevalent_colors}, (flat empty background)+++, masterpiece++, highly detailed+",
	# 	'caption_and_image_context': "darkest dungeon, (full body)+++ {entity_name}: ({entity_description})++, set in {place_description}, masterpiece++, highly detailed+"
	# }
	prompts = {
		'no_context': "(full body)+++ {entity_name}: ({entity_description})++, (flat empty background)+++, masterpiece++, highly detailed+",
		'colors_context': "(full body)+++ {entity_name}: ({entity_description})++, prevalent colors are {prevalent_colors}, (flat empty background)+++, masterpiece++, highly detailed+",
		'semantic_context': "(full body)+++ {entity_name}: ({entity_description})++, set in {place_name}: {place_description}, (flat empty background)+++, masterpiece++, highly detailed+",
		'semantic_and_colors_context': "(full body)+++ {entity_name}: ({entity_description})++, set in {place_name}: {place_description}, prevalent colors are {prevalent_colors}, (flat empty background)+++, masterpiece++, highly detailed+",
		'semantic_and_image_context': "(full body)+++ {entity_name}: ({entity_description})++, set in {place_name}: {place_description}, masterpiece++, highly detailed+",
		'caption_context': "(full body)+++ {entity_name}: ({entity_description})++, set in {place_description}, (flat empty background)+++, masterpiece++, highly detailed+",
		'caption_and_colors_context': "(full body)+++ {entity_name}: ({entity_description})++, set in {place_description}, prevalent colors are {prevalent_colors}, (flat empty background)+++, masterpiece++, highly detailed+",
		'caption_and_image_context': "(full body)+++ {entity_name}: ({entity_description})++, set in {place_description}, masterpiece++, highly detailed+"
	}
	entity_negative_prompt = "(close-up)+++, out of shot, (multiple characters)++, (many characters)++, not centered, floor, walls, pedestal, duplication, repetition, bad anatomy, monochromatic, disfigured, badly drawn, bad hands, naked, nude"

	prompt = prompts[context_level]
	formatting_values = {}

	entity_name, entity_description = clear_strings_for_prompt([entity_name, entity_description])
	# formatted_prompt = prompt.format(entity_name=entity_name, entity_description=entity_description)
	formatting_values['entity_name'] = entity_name
	formatting_values['entity_description'] = entity_description

	if room_name:
		formatting_values['place_name'] = clear_strings_for_prompt([room_name])
	if room_description:
		formatting_values['place_description'] = clear_strings_for_prompt([room_description])
	if colors_to_sd:
		colors_names = ' and '.join(colors_to_sd)
		formatting_values['prevalent_colors'] = clear_strings_for_prompt([colors_names])

	formatted_prompt = prompt.format(**formatting_values)

	log_message = f'ENTITY {context_level}: {entity_name} {entity_description}'
	with open(log_file, 'a') as f:
		f.write(f'{log_message} -> {formatted_prompt}\n')

	conditioning = compel_entity.build_conditioning_tensor(formatted_prompt)
	negative_conditioning = compel_entity.build_conditioning_tensor(entity_negative_prompt)
	[conditioning, negative_conditioning] = compel_entity.pad_conditioning_tensors_to_same_length(
		[conditioning, negative_conditioning])
	if entity_context_image is None:
		entity_image = sd_entity(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
		                         height=entity_image_size[0], width=entity_image_size[1],
		                         guidance_scale=sd_guidance_scale,
		                         num_inference_steps=entity_inference_steps,
		                         generator=th.Generator(device=device).manual_seed(base_rng_seed)).images[0]
		entity_image.save(f'./image_context_test_results/{exp_name}/{entity_name}_{room_name}_{context_level}_wb.png')
		entity_image = remove_background(entity_image)
	else:
		# control image is soft edges of context image
		control_image = entity_context_image.convert('L').filter(ImageFilter.FIND_EDGES)
		control_image.save(
			f'./image_context_test_results/{exp_name}/{entity_name}_{room_name}_control_{context_level}.png')
		# mask image is the alpha channel of context image
		_, _, _, mask_image = entity_context_image.split()
		# generate entity image
		entity_image = sd_inpaint_entity(
			prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning,
			num_inference_steps=entity_inference_steps,
			# eta=1.,
			guidance_scale=controlnet_guidance_scale,
			image=cropped_room_image,
			mask_image=mask_image,
			control_image=control_image,
			generator=th.Generator(device=device).manual_seed(base_rng_seed)).images[0]
		entity_image.save(f'./image_context_test_results/{exp_name}/{entity_name}_{room_name}_{context_level}_wb.png')
		# remove existing background using the mask image again
		entity_image_arr = np.array(entity_image)
		mask_arr = np.array(mask_image)
		entity_image_arr[mask_arr == 0, :] = 0
		entity_image_arr = np.concatenate([entity_image_arr, np.expand_dims(mask_arr, -1)], axis=-1)
		entity_image = PIL.Image.fromarray(entity_image_arr.astype(np.uint8))

	entity_image.save(f'./image_context_test_results/{exp_name}/{entity_name}_{room_name}_{context_level}.png')
	return entity_image


t = trange(n_runs * len(rooms) * len(enemies) * 8, desc='Generating images')

for n_run in range(n_runs):
	base_rng_seed = base_rng_seed + n_run
	exp_name = f"sd_{datetime.now().strftime('%Y%m%d%H%M%S')}"

	if not os.path.exists(f'./image_context_test_results/{exp_name}'):
		os.makedirs(f'./image_context_test_results/{exp_name}')
	log_file = f'./image_context_test_results/{exp_name}/logfile.log'

	with open(log_file, 'w') as f:
		f.write(f"{datetime.now().isoformat(timespec='microseconds')}\n")
		f.write(f'{max_n_colors=} {max_colors_to_sd=} {color_names=} {base_rng_seed=}\n\n')

	for i, (room_name, room_description) in enumerate(rooms.items()):
		# print('\t' + room_name + '...')
		t.set_postfix_str(f'{exp_name}; {room_name=} ({i + 1}/{len(rooms)})')
		room_image = generate_room(room_name, room_description)

		cropped_room_image = prepare_cropped_image(room_image)

		quantized_room_image = room_image.quantize(max_n_colors).convert('RGB')
		quantized_room_image.save(f'./image_context_test_results/{exp_name}/{room_name}_quantized_{max_n_colors}.png')
		all_colors = quantized_room_image.getcolors()
		all_colors_names = [convert_rgb_to_names(x[1]) for x in all_colors]
		all_colors_amounts = [x[0] for x in all_colors]
		colors_counts = Counter(all_colors_names)
		colors_to_sd = list(sorted(colors_counts.keys(), key=lambda x: colors_counts[1], reverse=True))[
		               :max_colors_to_sd]

		with open(log_file, 'a') as f:
			colors_info = [f'{n} ({ac[1]}, {ac[0]})' for n, ac in zip(all_colors_names, all_colors)]
			f.write(f'ALL COLORS: {" ".join(colors_info)}\n')
			f.write(f'COLORS TO SD: {" ".join(colors_to_sd)}\n')

		processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
		model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
		                                                     torch_dtype=th.float16).to("cuda")
		inputs = processor(room_image, return_tensors="pt").to("cuda", th.float16)
		out = model.generate(**inputs)
		generated_caption = processor.decode(out[0], skip_special_tokens=True)
		with open(log_file, 'a') as f:
			f.write(f'BLIP AUTO-CAPTION: {generated_caption}\n')

		for j, (entity_name, entity_description) in enumerate(enemies.items()):
			t.set_postfix_str(
				f'{exp_name}; {room_name=} ({i + 1}/{len(rooms)}); {entity_name=} ({j + 1}/{len(enemies)})')
			generate_entity(entity_name=entity_name, entity_description=entity_description, context_level='no_context',
			                room_name=room_name)
			t.n += 1
			t.refresh()
			generate_entity(entity_name=entity_name, entity_description=entity_description,
			                context_level='colors_context', room_name=room_name, colors_to_sd=colors_to_sd)
			t.n += 1
			t.refresh()
			entity_context_image = generate_entity(entity_name=entity_name,
			                                       entity_description=entity_description,
			                                       context_level='semantic_context', room_name=room_name,
			                                       room_description=room_description)
			t.n += 1
			t.refresh()
			generate_entity(entity_name=entity_name, entity_description=entity_description,
			                context_level='semantic_and_colors_context', room_name=room_name,
			                room_description=room_description, colors_to_sd=colors_to_sd)
			t.n += 1
			t.refresh()
			generate_entity(entity_name=entity_name, entity_description=entity_description,
			                context_level='semantic_and_image_context', room_name=room_name,
			                room_description=room_description,
			                entity_context_image=entity_context_image,
			                cropped_room_image=cropped_room_image)
			t.n += 1
			t.refresh()
			entity_context_image = generate_entity(entity_name=entity_name,
			                                       entity_description=entity_description,
			                                       context_level='caption_context', room_name=room_name,
			                                       room_description=generated_caption)
			t.n += 1
			t.refresh()
			generate_entity(entity_name=entity_name, entity_description=entity_description,
			                context_level='caption_and_colors_context', room_name=room_name,
			                room_description=generated_caption, colors_to_sd=colors_to_sd)
			t.n += 1
			t.refresh()
			generate_entity(entity_name=entity_name, entity_description=entity_description,
			                context_level='caption_and_image_context', room_name=room_name,
			                room_description=generated_caption,
			                entity_context_image=entity_context_image,
			                cropped_room_image=cropped_room_image)
			t.n += 1
			t.refresh()
