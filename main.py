from time import time

SPEED_MULTIPLIER = .5

sounds = {}
sounds['player'] = {'spawn': 'e10', 'death': 'e14'}
sounds['points'] = {'coins': 'e9', 'diamonds': 'e4'}
sounds['rage'] = {'activation': 'e11', 'opponent-kill': 'e18'}
sounds['open-gates'] = 'e21'
sounds['protection-end-animation'] = 'e8'
sounds['teleport'] = 'e19'
sounds['gameplay-music'] = 'm4'
sounds['game-over'] = 'm2'
sounds['game-finished'] = 'm3'

# 	Data Structures	
############################
class Node:
	def __init__(self, value):
		self.value = value
		self.next = None

class Queue:
	def __init__(self):
		self.first = None
		self.last = None
		self.length = 0
		
	def enqueue(self, value):
		new_node = Node(value)
		if self.is_empty():
			self.first = self.last = new_node
		else:
			self.last.next = new_node 
			self.last = new_node
		self.length += 1
		return new_node
		
	def dequeue(self):
		if self.is_empty():
			raise ValueError('No values remain in the queue')
		if self.first == self.last:
			self.last = None
		dequeued_node = self.first
		self.first = self.first.next
		self.length -= 1
		return dequeued_node.value
		
	def peek(self):
		if self.is_empty():
			return None
		return self.first.value
			
	def is_empty(self):
		return self.length == 0
		

#	Classes for Map Creation	
#####################################
class Block(Sprite):
	def __init__(self, properties, processed_properties=False):
        	self.properties = dict(properties)
		if not processed_properties:
			self.properties = self.__process_properties(properties)
		self.point_count = properties['point_count']
		
		self.image = properties['image']
		self.color = properties['color'] 
		self.size = properties['size']
		self.position = properties['position']
		self.angle = properties['angle']
		
	def __process_properties(self, properties):
		properties['color'] = self.__process_colors(properties.get('color', (0,0,0)))
		scale = self.__process_value(properties.get('scale', 1))
		properties['size'] *= scale
		properties['angle'] = self.__process_value(properties.get('angle', 0))
		if 'point_count' not in properties:
			properties['point_count'] = 0
		return properties
		
	def __process_value(self, value):
		if isinstance(value, tuple) or isinstance(value, list):
			value = self.__get_random_value(*value)
		return value
		
	def __process_colors(self, colors):
		if isinstance(colors, Color):
			return colors
		elif isinstance(colors[0], Color):
			return colors[random.randint(0, len(colors)-1)]
		elif isinstance(colors[0], tuple) or isinstance(colors[0], list):
			return Color(*colors[random.randint(0, len(colors)-1)])
		else:
			return Color(*colors)
	
	def __get_random_value(self, min_val, max_val):
		exp = max(self.__get_floating_length(min_val), self.__get_floating_length(max_val))
		mul = 10**exp
		random_value = random.randint(min_val * mul, max_val * mul) / float(mul)
		return random_value
		
	def __get_floating_length(self, value):
		return len(str(value-int(value))[2:])
	
	
class Plant(Block):
	def __init__(self, properties):
		Block.__init__(self, properties)
		self.max_angle = properties.get('max_angle', self.angle if self.angle != 0 else 5)
		self.delta_angle = .25
		self.animation_interval = .05
		self.last_animation_time = time()
		self.position_x = self.position.x
		self.position_y = self.position.y
		
	def update(self):
		self.sway()
		
	def sway(self):
		curr_time = time()
		if curr_time - self.last_animation_time >= self.animation_interval:
			if abs(self.angle) > self.max_angle:
				self.delta_angle = -self.delta_angle
			self.angle += self.delta_angle
			self.position.x = self.position_x - self.size / 2.0 * math.sin(self.angle * math.pi / 180)
			self.last_animation_time = curr_time
		
		
class Flame(Block):
	def __init__(self, properties):
		Block.__init__(self, properties)
		self.color = Color(*properties['colors'][0])
		
		# Burn animation
		self.max_angle = properties.get('max_angle', self.angle if self.angle != 0 else 5)
		self.curr_max_angle = random.randint(int(self.max_angle / 2), self.max_angle)
		self.curr_max_angle = 0
		self.block_size = properties['size']
		self.max_scale = properties['scale'][1]
		self.min_scale = properties['scale'][0]
		self.delta_size  = self.block_size * .01
		self.delta_angle = .8
		self.is_growing = False
		self.animation_interval = .05
		self.last_animation_time = time()
		self.position_x = self.position.x
		
		# Color animation
		self.color_1 = properties['colors'][0]
		self.color_2 = properties['colors'][1]
		self.colors_steps = [0,0,0] 		# red, green, blue
		self.color_anim_steps = random.randint(15, 40)
		self.current_anim_step = 0
		self.is_color_reversed = False
		
		self.__calc_light_anim_step()
		
	def update(self):
		curr_time = time()
		self.change_color(curr_time)
		self.burn(curr_time)
		
	def burn(self, curr_time):
		if curr_time - self.last_animation_time >= self.animation_interval:
			if abs(self.angle) > self.max_angle:
				self.delta_angle = -self.delta_angle
				self.curr_max_angle = random.randint(int(self.max_angle / 2), self.max_angle)
			self.angle += self.delta_angle
			self.position.x = self.position_x - self.size / 2.0 * math.sin(self.angle * math.pi / 180)
			if self.size < self.max_scale * self.block_size and self.is_growing:
				self.size += self.delta_size 
				self.position.y += self.delta_size / 2.0
			elif self.size > self.min_scale * self.block_size and not self.is_growing:
				self.size -= self.delta_size 
				self.position.y -= self.delta_size / 2.0
			else:
				self.is_growing = not self.is_growing
			self.last_animation_time = curr_time
			
	def change_color(self, curr_time):
		if curr_time - self.last_animation_time >= self.animation_interval:
			self.current_anim_step += 1
			if self.current_anim_step >= self.color_anim_steps:
				self.current_anim_step = 0
				self.is_color_reversed = not self.is_color_reversed
				self.color_anim_steps = random.randint(10, 25)
				self.__calc_light_anim_step()
			self.__change_light_color()
				
	def __calc_light_anim_step(self):
		self.colors_steps = [(c2 - c1) / self.color_anim_steps for c2, c1 in zip(self.color_1, self.color_2)]
			
	def __change_light_color(self):
		curr_color = self.color
		d_r, d_g, d_b = self.colors_steps
		if self.is_color_reversed:
			self.color = Color(curr_color.r + d_r, curr_color.g + d_g, curr_color.b + d_b)
		else:
			self.color = Color(curr_color.r - d_r, curr_color.g - d_g, curr_color.b - d_b)	
		
		
class Gate:
	def __init__(self, properties):
		self.properties = dict(properties)
		self.direction = properties['direction']
        	self.coords = ()
		self.gate_blocks = self.__create_gate()

	def spawn_gate(self):
		for gate_block in self.gate_blocks:
			game.add(gate_block)
		
	def remove_gate(self):
		for gate_block in self.gate_blocks:
			game.remove(gate_block)

    	def respawn_gate(self):
        	for i in range(len(self.gate_blocks)):
            		self.gate_blocks[i] = Block(self.gate_blocks[i].properties, True)
        	self.spawn_gate()
		
	def __create_gate(self):
		part_scale = self.properties.get('scale', 1)
		parts_count = round(1 / part_scale, 0)
		part_size = self.properties['size'] * part_scale
		vect = self.properties['position']
		move_vect = None
		gate_blocks = []
		if self.direction == 'horizontal':
			vect.x -= ((parts_count-1) / 2) * part_size
			move_vector = Vector(part_size, 0)
		elif self.direction == 'vertical':
			vect.y -= ((parts_count-1) / 2) * part_size
			move_vector = Vector(0, part_size)
		for i in range(parts_count):
			gate_part = Block(dict(self.properties))
			self.properties['position'] += move_vector
			gate_blocks.append(gate_part)
			
		return gate_blocks
		
		
class Rage(Block, Sprite):
	def __init__(self, properties, processed_properties=False):
		Block.__init__(self, properties, processed_properties)

		self.colors = properties['colors']
		self.duration = properties['duration']
		self.speed_boost = properties['speed_boost']
		
		self.color = Color(*self.colors[0])
		
		# Animations
		self.delta_time = 0.1
		self.last_anim_update_time = time()
		
		# Size animation
		self.max_size = self.size * 1.2
		self.min_size = self.size * .8
		self.delta_size = .05
		self.is_growing = True
		
		# Color animation
		self.color_steps = 40
		self.steps_counted = 10
		self.delta_color = [0, 0, 0] # red, green, blue
		
		self.next_color_idx = 0
		
	def update(self):
		self.handle_animation()
		
	def handle_animation(self):
		curr_time = time()
		if curr_time - self.last_anim_update_time >= self.delta_time:
			self.__next_anim_frame()
			
	def __next_anim_frame(self):
		self.__next_size_anim_frame()
		self.__next_color_anim_frame()
		
	def __next_size_anim_frame(self):
		if self.is_growing:
			if self.size < self.max_size:
				self.size += self.delta_size
			else:
				self.is_growing = False
		else:
			if self.size > self.min_size:
				self.size -= self.delta_size
			else:
				self.is_growing = True
	
	def __next_color_anim_frame(self):
		if self.steps_counted >= self.color_steps:
			self.steps_counted = 0
			curr_color = self.colors[self.next_color_idx]
			if self.next_color_idx + 1 < len(self.colors):
				self.next_color_idx += 1
			else:
				self.next_color_idx = 0
			next_color = self.colors[self.next_color_idx]
			self.delta_color = [(c2 - c1) / float(self.color_steps) for c2, c1 in zip(next_color, curr_color)]
		else:
			self.steps_counted += 1
			c = self.color
			d_r, d_g, d_b = self.delta_color
			self.color = Color(c.r + d_r, c.g + d_g, c.b + d_b)
			
		
class Teleport(Sprite):
	def __init__(self, properties):
		self.sprite_colors = properties['colors']
		self.max_sprite_size = properties['size'] * properties.get('scale', 1)
		self.sprite_image = properties['image']
		self.teleport_position = properties['position'] 		
		
		# For animation
		self.animation_steps = int(self.max_sprite_size) * 2
		self.delta_color = [0, 0, 0] # red, green, blue
		self.current_sprite_idx = 0
		self.last_anim_time = time()
		self.delta_time = .05
		self.is_anim_addding = False
		
		self.teleport_sprites = []
		self.num_sprites = 0
		self.create_teleport()
		
		self.size = .5
		self.position = self.teleport_position
		game.add(self)
		
	def update(self):
		self.handle_animation()
		
	def handle_animation(self):
		curr_time = time()
		if curr_time - self.last_anim_time >= self.delta_time:
			self.last_anim_time = curr_time
			for i in range(self.num_sprites):
				self.teleport_sprites[i].angle += i / 2.0
	
	def create_teleport(self):
		self.__update_colors()
		r, g, b = self.sprite_colors[0]
		d_r, d_g, d_b = self.delta_color
		for i in range(self.animation_steps, self.max_sprite_size / 2, -1):
			mul = (300.0/(i**3) * self.max_sprite_size)
			color = (r + d_r * mul, g + d_g * mul, b + d_b * mul)
			properties = { 'image': self.sprite_image, 'color': color, 'size': i / 2.0, 'angle' : i * 90.0 / self.animation_steps, 'position': self.teleport_position }
			teleport_sprite = Block(properties)
			self.teleport_sprites.append(teleport_sprite)
			self.num_sprites += 1
		self.current_sprite_idx = self.num_sprites-1

    	def recreate_teleport(self):
        	for i in range(len(self.teleport_sprites)):
            		self.teleport_sprites[i] = Block(self.teleport_sprites[i].properties, True)
			
	def load_teleport(self):
		for tp in self.teleport_sprites:
			game.add(tp)
			
	def remove_teleport(self):
		for tp in self.teleport_sprites:
			game.remove(tp)
			
	def __update_colors(self):
		self.delta_color = [c1 - c2 for c2, c1 in zip(*self.sprite_colors)]
		
		
class AnimatedKey(Block):
	def __init__(self, properties, partial_vectors):
		Block.__init__(self, properties)
		self.target_size = self.size
		self.size = 50
		self.position = Vector(0, 0)
		
		# Animation
		self.start_anim_time = self.last_frame_time = time()
		self.partial_vectors = partial_vectors
		self.delta_time = .03
		self.anim_duration = 2
		self.move_back_duration = .6
		self.delta_size = 0
		self.size_step = 0
		self.vect_step = None
		self.move_vect_coords = None
		self.move_dist = 150
		
		self.__process_anim_properties()
		
		game.add(self)
		
	def update(self):
		self.play_animation()
		
	def play_animation(self):
		curr_time = time()
		elapsed_time = curr_time - self.start_anim_time
		if curr_time - self.last_frame_time >= self.delta_time:
			self.last_frame_time = curr_time
			self.size += self.size_step
			if elapsed_time > self.move_back_duration:
				self.position += self.vect_step
		if 100 in [abs(self.position.x), abs(self.position.y)]:
			game.remove(self)
		
	def __process_anim_properties(self):
		self.move_vect_coords = self.__get_move_vector_coords()
		self.__update_anim_steps()
		
	def __get_move_vector_coords(self):
		d_x, d_y = self.partial_vectors
		if d_y == 0:
			if d_x > 0:
				return (self.move_dist, 0)
			else:
				return (-self.move_dist, 0)
		elif d_x == 0:
			if d_y > 0:
				return (0, self.move_dist)
			else:
				return (0, -self.move_dist)
		else: 
			return (self.move_dist * d_x, self.move_dist * d_y)
		
	def __update_anim_steps(self):
		move_back_steps_count = self.anim_duration / self.delta_time
		fly_out_steps_count = (self.anim_duration - self.move_back_duration) / self.delta_time
		v_x, v_y = self.move_vect_coords
		self.vect_step = Vector(float(v_x) / move_back_steps_count, float(v_y) / move_back_steps_count)
		self.size_step = (self.target_size - self.size) / fly_out_steps_count 
		
		
class Path:
	def __init__(self):
		self.collision_blocks_ids = []
		self.node_connections_dict = {}
		self.node_vectors_dict = {}
		self.nodes_banned_for_opponents_respawn = set()
		self.are_banned_nodes_updated = False
		self.max_opponent_coords_search_count = 20
		
	def set_collision_blocks(self, collision_blocks_ids):
		self.collision_blocks = collision_blocks_ids
		
	def add_node_to_path(self, map_arr, i, j, block_count_y, block_count_x, block_size):
		node_connections = self.__get_node_connections(map_arr, i, j, block_count_y, block_count_x)
		if node_connections:
			self.__add_to_connections_dict((j, i), node_connections)
			node_connections.add((j, i))
			self.__add_to_vectors_dict(node_connections, block_size)
		elif j in [0, block_count_x-1] or i in [0, block_count_y-1]:
			self.__add_to_banned_nodes((j, i))
			
	def get_all_connected_nodes(self, map_arr, i, j, block_count_y, block_count_x):
		adjacent_nodes = self.__check_adjacent_nodes(map_arr, i, j, block_count_y, block_count_x)
		is_right, is_left, is_up, is_down = adjacent_nodes
		connected_nodes = set()
		self.__update_connected_nodes(connected_nodes, adjacent_nodes, map_arr, i, j, block_count_y, block_count_x)
		return connected_nodes
		
	def get_nodes_banned_for_opponents(self):
		if not self.are_banned_nodes_updated:
			self.__update_banned_nodes()
		return self.nodes_banned_for_opponents_respawn
		
	def get_available_opponent_coords(self, current_coords, next_coords):
		banned_nodes = self.get_nodes_banned_for_opponents()
		nodes_dict = self.node_connections_dict
		if next_coords not in banned_nodes and current_coords not in banned_nodes:
			return True
		if next_coords in banned_nodes and next_coords in nodes_dict:
			current_coords = next_coords
		# Search for possible burrent coords
		node_check_queue = Queue()
		nodes_added_to_queue = set()
		connected_nodes = nodes_dict[current_coords]
		node_check_queue.enqueue(current_coords)
		while (current_coords in banned_nodes or next_coords in banned_nodes) and node_check_queue.length > 0:
			current_coords = node_check_queue.dequeue()
			connected_nodes = nodes_dict[current_coords]
			for node in connected_nodes:
				if current_coords not in banned_nodes and node not in banned_nodes:
					next_coords = node
					break
				elif node not in nodes_added_to_queue:
					node_check_queue.enqueue(node)
					nodes_added_to_queue.add(node)
		if next_coords not in connected_nodes:
			connected_nodes = list(connected_nodes)
			next_coords = connected_nodes[random.randint(0, len(connected_nodes)-1)]
				
		return (current_coords, next_coords)
	
	# ---------- Getters and Helpers ---------- 
	def __get_node_connections(self, map_arr, i, j, block_count_y, block_count_x):
		adjacent_nodes = self.__check_adjacent_nodes(map_arr, i, j, block_count_y, block_count_x)
		is_right, is_left, is_up, is_down = adjacent_nodes
		not_valid_node_vertical   = is_up and is_down and (not is_left and not is_right)
		not_valid_node_horizontal = is_left and is_right and (not is_up and not is_down)
		is_valid_node = not (not_valid_node_vertical or not_valid_node_horizontal)
		connected_nodes = set()
	
		if is_valid_node:
			self.__update_connected_nodes(connected_nodes, adjacent_nodes, map_arr, i, j, block_count_y, block_count_x)
				
		return connected_nodes
		
	def __check_adjacent_nodes(self, map_arr, i, j, block_count_y, block_count_x):
		is_right = map_arr[i][j+1] not in self.collision_blocks if j+1 < block_count_y else True
		is_left  = map_arr[i][j-1] not in self.collision_blocks if j-1 >= 0 else True
		is_up    = map_arr[i-1][j] not in self.collision_blocks if i-1 >= 0 else True
		is_down  = map_arr[i+1][j] not in self.collision_blocks if i+1 < block_count_x else True
		return [is_right, is_left, is_up, is_down]
		
	def __update_connected_nodes(self, connected_nodes, adjacent_nodes, map_arr, i, j, block_count_y, block_count_x):
		is_right, is_left, is_up, is_down = adjacent_nodes
		# Search for max right connected node
		if is_right: 
			k = j+1
			while k < block_count_x:
				if (map_arr[i][k] in self.collision_blocks or map_arr[i-1][k-1] not in self.collision_blocks or map_arr[i-1][k-1] not in self.collision_blocks) and k-1 != j:
					connected_nodes.add((k-1, i))
					break
				k += 1
			else:
				connected_nodes.add((block_count_x-1, i))
		# Search for max left connected node
		if is_left:
			k = j-1
			while k >= 0:
				if (map_arr[i][k] in self.collision_blocks or map_arr[i-1][k+1] not in self.collision_blocks or map_arr[i-1][k+1] not in self.collision_blocks) and k+1 != j:
					connected_nodes.add((k+1, i))
					break
				k -= 1
			else:
				connected_nodes.add((0, i))
		# Search for max down connected node
		if is_down:
			k = i+1
			while k < block_count_y:
				if (map_arr[k][j] in self.collision_blocks or map_arr[k-1][j-1] not in self.collision_blocks or map_arr[k-1][j+1] not in self.collision_blocks) and k-1 != i:
					connected_nodes.add((j, k-1))
					break
				k += 1
			else:
				connected_nodes.add((j, block_count_y-1))
		# Search for max up connected node
		if is_up:
			k = i-1
			while k >= 0:
				if (map_arr[k][j] in self.collision_blocks or map_arr[k+1][j-1] not in self.collision_blocks or map_arr[k+1][j+1] not in self.collision_blocks) and k+1 != i:
					connected_nodes.add((j, k+1))
					break
				k -= 1
			else:
				connected_nodes.add((j, 0))
				
	def __update_banned_nodes(self):
		for node in self.nodes_banned_for_opponents_respawn:
			self.nodes_banned_for_opponents_respawn = self.nodes_banned_for_opponents_respawn.union(self.node_connections_dict[node])
		self.are_banned_nodes_updated = True
	
	def __add_to_connections_dict(self, current_node, node_connections):
		# current_node is the new node added
		if current_node not in self.node_connections_dict:
			self.node_connections_dict[current_node] = set()
		self.node_connections_dict[current_node] = self.node_connections_dict[current_node].union(node_connections)
		#Set also current values as keys and add them current key node as value
		for checked_node in node_connections:
			if checked_node not in self.node_connections_dict:
				self.node_connections_dict[checked_node] = set()
			self.__remove_collinear_nodes(checked_node, current_node)

	def __remove_collinear_nodes(self, checked_node, current_node):
		current_connections = self.node_connections_dict[checked_node]
		# Remove further collinear vertical nodes 
		if checked_node[0] == current_node[0]:
			up_node   = list(filter(lambda c: c[1] < checked_node[1], current_connections))
			down_node = list(filter(lambda c: c[1] > checked_node[1], current_connections))
			# That means - added node is above the current node and another one above current node exists (decide which one to leave)
			if up_node and current_node[1] < checked_node[1]:
				if up_node[0][1] < current_node[1]:
					self.__replace_connected_node(checked_node, up_node[0], current_node)
			elif down_node and current_node[1] > checked_node[1]:
				if down_node[0][1] > current_node[1]:
					self.__replace_connected_node(checked_node, down_node[0], current_node)
			else:
				self.node_connections_dict[checked_node].add(current_node)
		else:
			# Remove further collinear horizontal nodes 
			right_node = list(filter(lambda c: c[0] > checked_node[0], current_connections))
			left_node  = list(filter(lambda c: c[0] < checked_node[0], current_connections))
			if right_node and current_node[0] > checked_node[0]:
				if right_node[0][0] > current_node[0]:
					self.__replace_connected_node(checked_node, right_node[0], current_node)
			elif left_node and current_node[0] < checked_node[0]:
				if left_node[0][0] < current_node[0]:
					self.__replace_connected_node(checked_node, left_node[0], current_node)
			else:
				self.node_connections_dict[checked_node].add(current_node)

	def __replace_connected_node(self, checked_node, removed_node, added_node):
		self.node_connections_dict[checked_node].discard(removed_node)
		self.node_connections_dict[checked_node].add(added_node)

	def __add_to_vectors_dict(self, node_connections, block_size):
		for node in node_connections:
			if node not in self.node_vectors_dict:
				i, j = node
				self.node_vectors_dict[node] = Vector(-100 + (i+.5) * block_size, 100 - (j+.5) * block_size)
				
	def __add_to_banned_nodes(self, coords):
		self.nodes_banned_for_opponents_respawn.add(coords)
		
	
class Board:
	# ---------- Necessary data ----------
	def __init__(self):
		self.blocks = {}
		self.blocks['block'] = { 'image': 63, 'scale': 1, 'color': (9, 158, 1) }
		self.blocks['bone'] = { 'image': 26, 'scale': (.4, .7), 'angle': (-180, 180), 'color': (255, 250, 227) }
		self.blocks['coin'] = { 'image': 55, 'scale': (.2, .3), 'color': (255, 247, 165) }
		self.blocks['diamond'] = { 'point_count': 5, 'image': 74, 'scale': .5, 'color': (62, 231, 255) }
		self.blocks['flame'] = { 'image': 48, 'scale': (.9, 1.25), 'colors': ((247, 70, 0), (247, 132, 0)), 'max_angle': 15 }
		self.blocks['gate'] =  { 'image': 63, 'scale': 1.0/3, 'color': (156, 149, 137) }
		self.blocks['home'] = { 'image': 78, 'scale': .8, 'color': (200, 200, 200) }
		self.blocks['key'] = { 'image': 79, 'scale': .6, 'color': (255, 187, 0) }
		self.blocks['mark'] = { 'image': 25, 'scale': (.3, .6), 'angle': (0, 180), 'color': (225, 160, 0) }
		self.blocks['oak'] = { 'image': 39, 'scale': (1, 1.35), 'color': (38, 190, 38) }
		self.blocks['rage'] = { 'duration': 5, 'speed_boost': 3.5, 'image': 55, 'scale': .4, 'colors': ((255, 0, 255), (0, 255, 255), (255, 255, 0)) }
		self.blocks['skull'] = { 'image': 21, 'scale': (.5, .8), 'angle': (-90, 90), 'color': (255, 250, 227) }
		self.blocks['spruce'] = { 'image': 38, 'scale': (1, 1.65), 'angle': (-5, 5), 'max_angle': 10, 'color': (4, 191, 116) }
		self.blocks['teleport'] = { 'image': 88, 'scale': 1, 'colors': ((130, 82, 2), (255, 80, 80)) }
		
		# Opponents styles
		self.blocks['ghost'] = { 'speed': 1, 'point_count': 20, 'image': 20, 'scale': .8, 'color': ((235, 139, 44), (235, 235, 44), (57, 194, 36), (35, 255, 161), (104, 115, 255), (226, 20, 169)) } 	
		self.blocks['pumpkin-ghost'] = { 'speed': 1.3, 'point_count': 30, 'image': 20, 'scale': .7, 'color': ((129, 8, 6), (191, 32, 14), (250, 65, 19), (254, 155, 19), (249, 193, 14)), 'helmet': { 'image': 19, 'color': ((249, 171, 70), (223, 91, 5), (251, 141, 19)), 'scale': .95 }, 'light': { 'image': 63, 'scale': .55, 'color': { 'on': (255, 255, 255), 'off': (0, 0, 0) } } } 	
		self.blocks['alien-ghost'] = { 'speed': 1.6, 'point_count': 40, 'image': 20, 'scale': .75, 'color': ((160, 210, 158), (145, 176, 157), (114, 148, 133), (182, 244, 133), (135, 212, 194), (168, 174, 222), (136, 105, 181)), 'helmet': { 'image': 22, 'color': (100,100,100), 'scale': 1 }, 'light': { 'image': 55, 'scale': .75, 'color': { 'on': (255, 0, 0), 'off': (0, 255, 255) } } } 	
		
		self.block_dict = {}
		self.block_dict['b'] = 'block'
		self.block_dict['d'] = 'diamond'
		self.block_dict['f'] = 'flame'
		self.block_dict['g'] = 'gate'
		self.block_dict['h'] = 'home'
		self.block_dict['k'] = 'key'
		self.block_dict['l'] = 'skull'
		self.block_dict['n'] = 'bone'
		self.block_dict['m'] = 'mark'
		self.block_dict['s'] = 'spruce'
		self.block_dict['r'] = 'rage'
		self.block_dict['t'] = 'teleport'
		self.block_dict['o'] = 'oak'
		self.block_dict['_'] = 'empty'
		self.block_dict['&'] = 'spawn'
		self.block_dict['x'] = 'blocked' # Empty block that nothing can appear on in (used in intro, outro, etc.)

		self.block_dict['1'] = 'ghost'
		self.block_dict['2'] = 'pumpkin-ghost'
		self.block_dict['3'] = 'alien-ghost'
		
		self.collision_blocks = ['s', 'o', 'b', 'f']
		
		self.boards = {}
		init_dict = lambda: {'name': '', 'segments_pattern': '', 'start_segment': {'segment_id': '', 'spawn_coords': '', 'spawn_vect': ''}, 'segments': {}, 'dimensions': {'x': 0, 'y': 0}, 'styles': {'background': '', 'blocks': {}}, 'keys_count': 0 }
		
		self.difficulty_levels = {}
		self.difficulty_levels['intro'] = 'intro'
		
		self.difficulty_levels[1] = 'easy'
		self.difficulty_levels[2] = 'medium'
		self.difficulty_levels[3] = 'hard'
		
		self.difficulty_levels['lose'] = 'game-over'
		self.difficulty_levels['win'] = 'win'
		
		for lvl_name in self.difficulty_levels.values():
			self.boards[lvl_name] = init_dict()
		
		self.functions_dict = {}
		self.functions_dict['not'] = {} # Functions called for all other blocks than specified
		self.functions_dict['key'] = lambda: self.__process_key()
		self.functions_dict['gate'] = lambda: self.__process_gate()
		self.functions_dict['rage'] = lambda: self.__process_rage()
		self.functions_dict['spawn'] = lambda: self.__process_spawn()
		self.functions_dict['diamond'] = lambda: self.__process_diamond()
		self.functions_dict['teleport'] = lambda: self.__process_teleport()
		self.functions_dict[('empty', 'mark', 'bone', 'skull')] = lambda: self.__add_to_coin_spawn_points()
		self.functions_dict[('block', 'mark', 'home', 'bone', 'skull')] = lambda: self.__process_block()
		self.functions_dict['flame'] = lambda: [self.__process_block('block'), self.__process_flame()]
        	self.functions_dict[('spruce', 'oak')] = lambda: [self.__process_block('block'), self.__process_plant()]
		self.functions_dict['ghost'] = lambda: self.__process_opponent(Ghost, 'ghost', (self.j, self.i), False) 
		self.functions_dict['alien-ghost'] = lambda: self.__process_opponent(AlienGhost, 'alien-ghost', (self.j, self.i), True) 
		self.functions_dict['pumpkin-ghost'] = lambda: self.__process_opponent(PumpkinGhost, 'pumpkin-ghost', (self.j, self.i), True) 
		self.functions_dict['not'][('block', 'spruce', 'oak', 'flame', 'blocked')] = lambda: self.__add_node_to_path()
	
	# ---------- Initialisation ----------
	def init_level(self, diff_lvl):
		curr_lvl = self.difficulty_levels[diff_lvl]
		self.current_difficulty_level = diff_lvl
		self.current_segment_id = self.boards[curr_lvl]['start_segment']['segment_id']
		self.remain_keys_num = self.boards[curr_lvl]['keys_count']
		self.current_points = { 'coins': [], 'diamonds': [] }
		self.current_coin_spawn_points = []
		self.current_block_sprites = [] 
		self.current_moving_sprites = []	
		self.opponents_respawn_queue = Queue()
		self.current_opponents = []
		self.current_gates = []
		self.current_teleport = None
		self.lvl_unlock_key = None
		self.current_path = None
		self.current_rage = None
		self.current_key = None
		self.cached_blocks = {}
		self.block_size = 0
		
	def set_map_pattern(self, diff_lvl, name, segments_str, start_segment, x_size=20, y_size=20):
		curr_map = self.boards[diff_lvl]
		curr_map['name'] = name
		curr_map['segments_pattern'] = self.clear_string(segments_str)
		curr_map['start_segment']['segment_id'] = start_segment
		curr_map['dimensions']['x'] = x_size
		curr_map['dimensions']['y'] = y_size
	
	def set_map_background(self, diff_lvl, color):
		self.boards[diff_lvl]['styles']['background'] = Color(*color)
			
	def set_map_styles(self, diff_lvl, block_name, style_dict):
		self.boards[diff_lvl]['styles']['blocks'][block_name] = style_dict
		
	def add_map_segment(self, diff_lvl, seg_id, map_str):
		curr_map = self.boards[diff_lvl]
		map_dimensions = curr_map['dimensions']
		rows = self.clear_string(map_str)
		# Validate dimensions of the added segment 
		if (len(rows[0]) != map_dimensions['x'] or len(rows) != map_dimensions['y']):
			raise Exception('Wrong dinemsions of added segment.')
		curr_map['segments'][seg_id] = rows 
		if filter(lambda row: 'k' in row, rows):
			curr_map['keys_count'] += 1
		
	# ---------- Map loading and others ----------
	def load_map(self):
		# Load previously loaded map segment from cache if exists
		if self.current_segment_id in self.cached_blocks: 
			self.__load_cached_map()
		else: 
			self.__build_map() # Else build map from scratch
			self.__spawn_coins()
		
	def clear_map(self):
		# Cache data
		self.__update_cache()
		# Remove block Sprites from map
		self.opponents_respawn_queue = Queue()
		if self.lvl_unlock_key:
			game.remove(self.lvl_unlock_key)
		if self.current_key:
			game.remove(self.current_key)
			self.current_key = None
		if self.current_rage:
			game.remove(self.current_rage)
			self.current_rage = None
		if self.current_teleport:
			self.current_teleport.remove_teleport()
			self.current_teleport = None
		while self.current_opponents:
			self.current_opponents.pop().remove_opponent()
		while self.current_block_sprites:
			game.remove(self.current_block_sprites.pop())
		while self.current_moving_sprites:
			game.remove(self.current_moving_sprites.pop())
		while self.current_points['diamonds']:
			game.remove(self.current_points['diamonds'].pop())
		while self.current_points['coins']:
			game.remove(self.current_points['coins'].pop())
		while self.current_gates:
			self.current_gates.pop().remove_gate()
			
	def open_segment_gates(self):
        	sound.play(sounds['open-gates'])
        	self.player.gates_coords = []
		while self.current_gates:
			gate = self.current_gates.pop()
			gate.remove_gate()
			
	def reload_to_segment(self, seg_id):
		seg_id = str(seg_id)
		if seg_id != self.current_segment_id:
			curr_lvl = self.difficulty_levels[self.current_difficulty_level]
			self.clear_map()
			self.current_segment_id = seg_id
			self.load_map()
			
	def add_block_to_game(self, block_sprite):
		self.current_block_sprites.append(block_sprite)
		game.add(block_sprite)
		
	def add_moving_block_to_queue(self, moving_block_sprite):
		self.current_moving_sprites.append(moving_block_sprite)
		
	def spawn_moving_sprites(self):
		for moving_sprite in self.current_moving_sprites:
			game.add(moving_sprite)
			
	def respawn_moving_sprites(self):
		for i in range(len(self.current_moving_sprites)):
			removed_sprite = self.current_moving_sprites[i]
			Cls = eval(removed_sprite.__class__.__name__)
            		game.remove(removed_sprite)
			self.current_moving_sprites[i] = Cls(removed_sprite.properties)
			self.current_moving_sprites[i].size = removed_sprite.size
			self.current_moving_sprites[i].position = removed_sprite.position 
			self.current_moving_sprites[i].angle = removed_sprite.angle 
			game.add(self.current_moving_sprites[i])
			
	def create_opponent(self, Cls, opponent_name, coords, gates_are_obstacles):
		properties = { 'name': opponent_name, 'spawn_coords': coords, 'position': self.get_vector(coords), 'size': self.block_size, 'gates_are_obstacles': gates_are_obstacles }
		properties = self.__get_updated_block_properties(opponent_name, properties)
		opponent = Cls(self.current_path, properties)
		if coords not in self.current_path.node_connections_dict.keys():
			possible_next_nodes = self.current_path.get_all_connected_nodes(self.map_arr, self.i, self.j, self.block_count_y, self.block_count_x)
			opponent.next_coords = list(possible_next_nodes)[random.randint(0, len(possible_next_nodes)-1)]
			opponent.update_move_vector()
		self.current_opponents.append(opponent)
		return opponent
			
	def spawn_opponents(self):
		for opponent in self.current_opponents:
			if opponent.gates_are_obstacles:
				opponent.current_obstacles = self.get_gate_blocks()
			# Move opponent one block back if it is on the segment bound
			opponent.spawn_opponent()
	
	# ---------- Getters and helpers ----------
	def get_gate_blocks(self):
		return [block for gate in self.current_gates for block in gate.gate_blocks]
	
	def get_vector(self, coords):
		j, i = coords
		return Vector(-100 + (j+.5) * self.block_size, 100 - (i+.5) * self.block_size)
	
	def get_block(self, block_name, prop_dict = {}):
		properties = self.__get_updated_block_properties(block_name, prop_dict)
		return Gate(properties) if block_name == 'gate' else Block(properties)
		
	def get_plant(self, plant_name, block_size, prop_dict = {}):
		properties = self.__get_updated_block_properties(plant_name, prop_dict)
		plant = Plant(properties)
		delta_size = (plant.size - block_size) / 2.0
		plant.position.x += random.randint(-delta_size, delta_size)
		plant.position.y += delta_size
		return plant	
		
	def get_flame(self, block_name, block_size, prop_dict = {}):
		properties = self.__get_updated_block_properties(block_name, prop_dict)
		flame = Flame(properties)
		delta_y = flame.size / 2.0
		flame.position.y += delta_y
		return flame
		
	def __get_updated_block_properties(self, block_name, prop_dict):
		curr_lvl = self.difficulty_levels[self.current_difficulty_level]
		curr_board = self.boards[curr_lvl]
		properties = {}
		properties.update(self.blocks[block_name])
		properties.update(curr_board['styles']['blocks'].get(block_name, ''))
		properties.update(prop_dict)	
		return properties
		
	def __get_gate_direction(self, map_arr, block_count_x, i, j):
		solid_blocks = [self.block_dict[id] for id in self.collision_blocks]
		if j-1 >= 0 and j+1 < block_count_x:
			left_block = self.block_dict[map_arr[i][j+1]]
			right_block = self.block_dict[map_arr[i][j-1]]
			if left_block in solid_blocks and right_block in solid_blocks:
				return 'horizontal'
		return 'vertical'
		
	def clear_string(self, string):
		out_str = []
		for row in string.splitlines():
			splitted_row = ' '.join(row.split()).split(');')
			splitted_val = ''
			if len(splitted_row) > 1:
				splitted_val = splitted_row[1].split('\n')[0]
			elif len(splitted_row[0]) > 1:
				splitted_val = splitted_row[0].split('\n')[0]
			if not splitted_val.isspace() and not splitted_val == '':
				out_str.append(splitted_val)
		return out_str

	# ---------- Map buildiers ----------
	def __build_map(self):
		diff_lvl = self.difficulty_levels[self.current_difficulty_level]
		self.curr_map = self.boards[diff_lvl]
		dimensions = self.curr_map['dimensions']
		self.block_count_x = dimensions['x']
		self.block_count_y = dimensions['y'] 
		self.map_arr = self.curr_map['segments'][self.current_segment_id]
		self.block_size = 200.0 / max(self.block_count_x, self.block_count_y)
		self.current_path = Path()
		self.current_path.set_collision_blocks(self.collision_blocks)
		#self.current_coins = Coins()
		game.background = self.curr_map['styles']['background']
		# Fill board with block objects
		for i in range(self.block_count_y):
			for j in range(self.block_count_x):
				self.i = i
				self.j = j
				self.__process_current_block()
			
	def __process_current_block(self):
		block_id = self.map_arr[self.i][self.j]
		self.vect = Vector(-100 + (self.j+.5) * self.block_size, 100 - (self.i+.5) * self.block_size)
		self.block_name = self.block_dict.get(block_id, 'empty')
		self.curr_properties = { 'position': self.vect, 'size': self.block_size }
	
		# Call proper function for current block
		for block_names, func in self.functions_dict.items():
			if block_names == 'not':
				continue
			if (self.block_name in block_names and isinstance(block_names, tuple) or isinstance(block_names, list)) or self.block_name == block_names:
				self.__call_functions(func)
			
		# Call functions which execute for all blocks beyond specified
		for block_names, func in self.functions_dict['not'].items():
			if self.block_name not in block_names:
				self.__call_functions(func())
				
	def __process_key(self):
		key = self.get_block(self.block_name, self.curr_properties)
		self.current_key = key 
		game.add(key)
	
	def __process_gate(self):
		direction = self.__get_gate_direction(self.map_arr, self.block_count_x, self.i, self.j)
		self.curr_properties['direction'] = direction
		block = self.get_block('gate', self.curr_properties)
		block.coords = (self.j, self.i)
		self.current_gates.append(block)
		block.spawn_gate()
		
	def __process_plant(self):
		plant = self.get_plant(self.block_name, self.block_size, self.curr_properties)
		self.add_moving_block_to_queue(plant)
		
	def __process_block(self, block_name=None):
		block = self.get_block(block_name if block_name else self.block_name, self.curr_properties)
		self.add_block_to_game(block)
	
	def __process_spawn(self):
		self.curr_map['start_segment']['spawn_coords'] = (self.j, self.i)
		self.curr_map['start_segment']['spawn_vect'] = self.vect
		
	def __process_diamond(self):
		diamond = self.get_block(self.block_name, self.curr_properties)
		self.current_points['diamonds'].append(diamond)
		game.add(diamond)
		
	def __process_opponent(self, Cls, opponent_name, coords, gates_are_obstacles=False):
		self.create_opponent(Cls, opponent_name, coords, gates_are_obstacles)
		
	def __process_rage(self):
		properties = self.__get_updated_block_properties(self.block_name, self.curr_properties)
		self.current_rage = Rage(properties)
		game.add(self.current_rage)
		
	def __process_teleport(self):
		properties = self.__get_updated_block_properties(self.block_name, self.curr_properties)
		self.current_teleport = Teleport(properties)
		self.current_teleport.load_teleport()
		
	def __process_flame(self):
		self.__process_block('block')
		flame = self.get_flame(self.block_name, self.block_size, self.curr_properties)
		self.add_moving_block_to_queue(flame)
		
	def __spawn_coins(self):
		spawn_points = self.current_coin_spawn_points
		max_coins_count = len(spawn_points)
		spawned_coins_count = random.randint(max_coins_count / 2, max_coins_count * .9)
		coin_size_bounds = [self.block_size * n for n in self.blocks['coin']['scale']]
		for i in range(spawned_coins_count):
			rand_num = random.randint(0, len(self.current_coin_spawn_points)-1)
			vect = spawn_points.pop(rand_num)
			properties = { 'position': vect, 'size': self.block_size }
			coin = self.get_block('coin', properties)
			coin_pts = 2 if coin_size_bounds[1] - coin.size < coin.size - coin_size_bounds[0] else 1
			coin.point_count = coin_pts
			self.current_points['coins'].append(coin)
			game.add(coin)
		self.current_coin_spawn_points = []
		
	def __add_to_coin_spawn_points(self):
		self.current_coin_spawn_points.append(self.vect)
		
	def __add_node_to_path(self):
		self.current_path.add_node_to_path(self.map_arr, self.i, self.j, self.block_count_y, self.block_count_x, self.block_size)
	
	def __call_functions(self, functions):
		if isinstance(functions, list) or isinstance(functions, tuple):
			for func in functions:
				if func:
					func()
		elif functions:
			functions()
			
	# ---------- Memoization ----------
	def __update_cache(self):
		segment_id = self.current_segment_id
		if segment_id not in self.cached_blocks:
			# Data cached only once
			self.cached_blocks[segment_id] = {}
			cached = self.cached_blocks[segment_id]
			cached['points'] = {}
			cached['moving-blocks'] = self.current_moving_sprites[:]
			cached['blocks'] = self.current_block_sprites[:]
			cached['path'] = self.current_path
			cached['teleport'] = self.current_teleport
			cached['unlock-key'] = None
		# Data cached on every reload
		cached = self.cached_blocks[segment_id]
		cached['key'] = self.current_key
		cached['rage'] = self.current_rage
		cached['gates'] = self.current_gates[:]
		cached['points']['coins'] = self.current_points['coins'][:]
		cached['points']['diamonds'] = self.current_points['diamonds'][:]
		cached['opponents'] = self.current_opponents[:]
		cached['opponents-to-respawn'] = self.opponents_respawn_queue
		
	def __load_cached_map(self):
		# Add data to board current state
		cached = self.cached_blocks[self.current_segment_id]
		self.current_path = cached['path']
		self.current_gates = cached['gates'][:]
		self.current_opponents = cached['opponents'][:]
		self.current_moving_sprites = cached['moving-blocks'][:]
		self.opponents_respawn_queue = cached['opponents-to-respawn']
		
        	# Load Sprites to re-create the map
       		# Recreation of the sprites is required on the grounds that PixBlocks starts lagging while using the same instance more than one time (bug?)
		for gate in self.current_gates:
			gate.respawn_gate()
		for block in cached['blocks']:
            		block = Block(block.properties, True)
			self.add_block_to_game(block)
		for i in range(len(cached['points']['diamonds'])):
            		diamond = Block(cached['points']['diamonds'][i].properties, True)
			self.current_points['diamonds'].append(diamond)
            		game.add(diamond)
		for i in range(len(cached['points']['coins'])):
			coin = Block(cached['points']['coins'][i].properties, True)
			self.current_points['coins'].append(coin)
            		game.add(coin)	
		if cached['key']:
            		self.current_key = Block(cached['key'].properties, True)
			game.add(self.current_key)
        	if cached['unlock-key']:
            		self.lvl_unlock_key = Block(cached['unlock-key'].properties, True)
            		game.add(self.lvl_unlock_key)
		if cached['rage']:
            		self.current_rage = Rage(cached['rage'].properties, True)
			game.add(self.current_rage)
		if cached['teleport']:
            		self.current_teleport = cached['teleport']
            		self.current_teleport.recreate_teleport()
			self.current_teleport.load_teleport()
		
#	Player and Opponents Classes		
##########################################
class Player(Sprite):
	def __init__(self, curr_coords, current_path, angle, block_size):
		# Properties referring to the map
		self.block_size = block_size
		self.current_path = current_path
		self.prev_coords = curr_coords
		self.current_coords = curr_coords
		self.next_coords = None
		self.current_obstacles = None
        	self.gates_coords = None
		self.next_angle = angle
		self.block_count_x = 0
		self.block_count_y = 0

		# Sprite properties
		self.position = self.__get_initial_vector()
		self.image = 60
		self.color = Color(255, 207, 0)
		self.size = block_size * .8
		self.angle = angle
		
		# For animations
		self.time_interval = .05
		self.last_anim_update_time = time()
		self.default_color = Color(255, 207, 0)
		self.rage_color = Color(255, 35, 0)	
		self.death_color = Color(140, 140, 140)
		self.play_death_anim = False
		
		# Respawn
		self.death_protection = False
		self.death_animation_started = False
		self.death_protection_time = 0
		self.death_protection_start_time = 0
		
		# Rage mode
		self.rage_mode = False
		self.default_speed = 2 * SPEED_MULTIPLIER
		self.rage_speed = 0
		self.last_speed_update_time = 0
		self.last_color_update_time = 0
		
		# Movement
		self.is_stopped = False
		self.speed = self.default_speed
		self.can_turn = [False, False, False, False] # Right, Left, Up, Down
		self.collide_gate = False
        	self.__update_possible_turns()
		
	def update(self):
		self.play_animation(.3)
		if not self.is_stopped:
			self.handle_movement()
			self.handle_gate_collision() 
			self.handle_rage_mode() 
		
	# ---------- Used in .update() method ----------
	def play_animation(self, delay):
		curr_time = time()
		delay *= self.default_speed / float(self.speed)  
		if curr_time - self.last_anim_update_time >= delay:
			self.image = 60 if self.image == 55 else 55
			self.last_anim_update_time = curr_time
			if self.death_protection:
				if not self.death_animation_started:
					self.death_protection_start_time = curr_time	
				self.death_animation_started = True
				death_c = self.death_color
				curr_c = self.color
				if death_c.r == curr_c.r and death_c.g == curr_c.g and death_c.b == curr_c.b:
					self.color = self.default_color
				else:
		                    	sound.play(sounds['protection-end-animation'])
		                    	self.color = self.death_color
			if curr_time - self.death_protection_start_time > self.death_protection_time:
				self.death_protection = False
			
	def handle_movement(self):
		# Get user input and process when possible to turn the player's direction
		if game.key('w') or game.key('up'):
			self.next_angle = 90
		elif game.key('s') or game.key('down'):
			self.next_angle = -90
		elif game.key('d') or game.key('right'):
			self.next_angle = 0
		elif game.key('a') or game.key('left'):
			self.next_angle = 180
		
		i, j = self.current_coords
		# Get coordinates of node which player is heading to 
		if self.can_turn[0] and self.angle == 0:
			self.next_coords = (i+1, j)
			self.move(self.speed)
		elif self.can_turn[1] and self.angle == 180:
			self.next_coords = (i-1, j)
			self.move(self.speed)
		elif self.can_turn[2] and self.angle == 90:
			self.next_coords = (i, j-1)
			self.move(self.speed)
		elif self.can_turn[3] and self.angle == -90:
			self.next_coords = (i, j+1)
			self.move(self.speed)
		
		self.update_coords()
		
	def handle_gate_collision(self):
		if self.collide(self.current_obstacles) or self.next_coords in self.gates_coords:
            		if self.angle == 0 and (game.key('a') or game.key('left')):
				self.angle = 180
				self.move(self.speed)
				self.__swap_coords()
			elif self.angle == 180 and (game.key('d') or game.key('right')):
				self.angle = 0
				self.move(self.speed)
				self.__swap_coords()
			elif self.angle == 90 and (game.key('s') or game.key('down')):
				self.angle = -90
				self.move(self.speed)
				self.__swap_coords()
			elif self.angle == -90 and (game.key('w') or game.key('up')):
				self.angle = 90
				self.move(self.speed)
				self.__swap_coords()
			elif not self.collide_gate:
                		self.move(-self.speed + (self.block_size - self.current_obstacles[0].size)*.75)
                		self.collide_gate = True
            		else:
				self.move(-self.speed)

			self.__update_possible_turns()
        	else:
            		self.collide_gate = False
				
	def handle_rage_mode(self):
		if self.rage_mode:
			self.speed = self.rage_speed
			self.color = self.rage_color
			self.last_speed_update_time = time()
		else:
			if self.speed > self.default_speed:
				curr_time = time()
				if curr_time - self.last_speed_update_time >= self.time_interval:
					self.last_speed_update_time = curr_time
					self.speed *= .97
				if curr_time - self.last_color_update_time >= self.time_interval * 5:
					self.last_color_update_time = curr_time
					has_default_color = self.color.r == self.default_color.r and self.color.g == self.default_color.g and self.color.b == self.default_color.b
					if has_default_color:
						self.color = self.rage_color
					else: 
                        			sound.play(sounds['protection-end-animation'])
						self.color = self.default_color		
			else:
				self.speed = self.default_speed
				if not self.death_protection:
					self.color = self.default_color

    	def update_coords(self):
		# Get vector of next coordinates and update to current coords when reached
		next_coords_vect = self.__get_vector(self.next_coords)
		curr_position_vect = self.position
        
		if (next_coords_vect - curr_position_vect).length <= self.speed and self.next_coords not in self.gates_coords:
			self.prev_coords, self.current_coords = self.current_coords, self.next_coords
			self.__update_possible_turns()
			self.position = self.__get_vector(self.current_coords)
			if self.can_turn[0] and self.next_angle == 0:
				self.angle = self.next_angle
			elif self.can_turn[1] and self.next_angle == 180:
				self.angle = self.next_angle
			elif self.can_turn[2] and self.next_angle == 90:
				self.angle = self.next_angle
			elif self.can_turn[3] and self.next_angle == -90:
				self.angle = self.next_angle
			self.next_angle = None
		
	# ---------- Used outside this class ----------
	def update_obstacles(self, obstacles):
		self.current_obstacles = obstacles
		
	def update_map_bounds(self, block_count_x, block_count_y):
		self.block_count_x = block_count_x
		self.block_count_y = block_count_y
		
	def stop_moving(self):
		self.is_stopped = True
		
	def start_moving(self):
		self.is_stopped = False

	# ---------- Helpers and getters ----------
	def __get_vector(self, coords):
		i, j = coords
		return Vector(-100 + (i+.5) * self.block_size, 100 - (j+.5) * self.block_size)
		
	def __get_next_possible_coords(self):
		next_possible_coords = self.current_path.node_connections_dict.get(self.current_coords, [])
		return next_possible_coords
		
	def __swap_coords(self):
		self.current_coords, self.next_coords = [self.next_coords, self.current_coords]
		self.prev_coords = self.current_coords
		
	def __get_initial_vector(self):
		if not self.current_coords in self.current_path.node_vectors_dict.keys():
			# Get closest coordinates to these specified
			curr_coords = self.current_coords
			last_distance = 200
			closest_coords = curr_coords
			for coords in self.current_path.node_vectors_dict.keys():
				curr_distance = self.__calc_distance(curr_coords, coords)
				if curr_distance < last_distance:
					last_distance = curr_distance
					closest_coords = coords
			self.current_coords = closest_coords
		return self.current_path.node_vectors_dict[self.current_coords]
				
	def __calc_distance(self, coords_1, coords_2):
		return math.sqrt((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2)
		
	def __update_possible_turns(self):
		next_possible_coords = self.__get_next_possible_coords()
		i, j = self.current_coords
			
		# Decide in which directions can player go from current node
		if not next_possible_coords:
			if self.angle in (0, 180):
				self.can_turn = [True, True, False, False]
			elif self.angle in (-90, 90):
				self.can_turn = [False, False, True, True]
		else:
			self.can_turn[0] = True if filter(lambda coords: coords[0] >  i and coords[1] == j, next_possible_coords) else False # Right
			self.can_turn[1] = True if filter(lambda coords: coords[0] <  i and coords[1] == j, next_possible_coords) else False # Left 
			self.can_turn[2] = True if filter(lambda coords: coords[0] == i and coords[1] <  j, next_possible_coords) else False # Up
			self.can_turn[3] = True if filter(lambda coords: coords[0] == i and coords[1] >  j, next_possible_coords) else False # Down
			# Fix for map bounds
			if i in (0, self.block_count_x-1):
				self.can_turn[0] = self.can_turn[1] = True
			elif j in (0, self.block_count_y-1):
				self.can_turn[2] = self.can_turn[3] = True
	
	
class Opponent(Block):
	def __init__(self, curr_path, properties):
		Block.__init__(self, properties)
		self.name = properties['name']
		self.speed = properties['speed'] * SPEED_MULTIPLIER
		self.start_coords = properties['spawn_coords']
		
		self.gates_are_obstacles = properties['gates_are_obstacles']
		self.current_obstacles = None
		self.turn_back_probability = 1.0/6
		
		self.move_interval = .02
		self.last_move_time = time()
		self.current_path = curr_path
		self.current_coords = self.start_coords
		self.next_coords = self.current_coords
		self.prev_coords = self.current_coords
		self.move_vector = Vector(0, 0)
	
		# Rage mode
		self.rage_mode = False
		self.rage_speed = 0	
		self.rage_color = Color(89, 97, 255)
		self.default_speed = self.speed
		self.default_color = self.color
		
		# Other functionalities
		self.is_stopped = False
	
	def update(self):
		if time() - self.last_move_time >= self.move_interval and not self.is_stopped:
			self.last_move_time = time()
			self.__handle_rage_mode()	
			self.__handle_movement()
			self.__handle_collisions()
			
	# ---------- Used outside this class ----------
	def update_move_vector(self):
		self.__update_opponent_move_vector()
	
	def update_movement(self):
		self.__update_opponent_movement()
		
	def spawn_opponent(self):
		game.add(self)
		
	def remove_opponent(self):
		game.remove(self)
		
	def stop_moving(self):
		self.is_stopped = True
		
	def start_moving(self):
		self.is_stopped = False
		
	# ---------- Used only inside this class ----------
	def __handle_movement(self):
		self.__trace_opponent_position()
		self.position += self.move_vector
		
	def __handle_collisions(self):
		x = self.position.x
		y = self.position.y

		if abs(x) >= 95 or abs(y) >= 95:
			self.__turn_back()
		if self.gates_are_obstacles and self.current_obstacles:
			if self.collide(self.current_obstacles):
				self.__turn_back()
				
	def __turn_back(self):
		# Make opponent move back by swapping coordinates and reversing movement vector
		self.next_coords, self.current_coords = [self.current_coords, self.next_coords]
		i, j = self.next_coords
		# Search for next possible coords
		while (i, j) not in self.current_path.node_connections_dict:
			if self.next_coords[1] == self.current_coords[1]:
				if self.next_coords[0] > self.current_coords[0]:
					i += 1
				else:
					i -= 1
			elif self.next_coords[0] == self.current_coords[0]:
				if self.next_coords[1] > self.current_coords[1]:
					j -= 1
				else:
					j += 1
		self.next_coords = (i, j)
		self.move_vector = Vector(-self.move_vector.x, -self.move_vector.y)
		if self.rage_mode:
			self.move_vector = Vector(-self.move_vector.x * 1.25, -self.move_vector.y * 1.25)
		
	def __handle_rage_mode(self):
		if self.rage_mode:
			self.speed = self.rage_speed * .75
			self.__update_opponent_move_vector()
			self.color = self.rage_color
		else:
			self.speed = self.default_speed
			self.color = self.default_color
			self.__update_opponent_move_vector()
			
	def __get_next_possible_coords(self):
		next_possible_coords = self.current_path.node_connections_dict.get(self.current_coords, [])
		return next_possible_coords
		
	def __get_next_random_coords(self):
		next_possible_coords = list(self.__get_next_possible_coords())
		num_possible_coords = len(next_possible_coords)
		can_move_back = (0 == random.randint(0, int(1.0 / self.turn_back_probability - 1))) 
		if num_possible_coords > 1 and self.prev_coords in next_possible_coords and not can_move_back:
			next_possible_coords.remove(self.prev_coords)
			num_possible_coords -= 1
		random_coords = next_possible_coords[random.randint(0, num_possible_coords-1)]
		return random_coords
	
	def __trace_opponent_position(self):
		next_coords_vect = self.current_path.node_vectors_dict[self.next_coords]
		curr_position_vect = self.position
		
		if (next_coords_vect - curr_position_vect).length <= self.speed:
			self.__update_opponent_movement()
			
	def __update_opponent_movement(self):
		self.prev_coords = self.current_coords 
		self.current_coords = self.next_coords
		self.position = self.current_path.node_vectors_dict[self.current_coords]
		self.next_coords = self.__get_next_random_coords()
		self.__update_opponent_move_vector()
			
	def __update_opponent_move_vector(self):
		if self.next_coords[0] == self.current_coords[0]:
			if self.next_coords[1] > self.current_coords[1]:
				self.move_vector = Vector(0, -self.speed)
			else:
				self.move_vector = Vector(0, self.speed)
		elif self.next_coords[1] == self.current_coords[1]:
			if self.next_coords[0] > self.current_coords[0]:
				self.move_vector = Vector(self.speed, 0)
			else:
				self.move_vector = Vector(-self.speed, 0)
	
	
class Ghost(Opponent):
	def __init__(self, curr_path, properties):
		Opponent.__init__(self, curr_path, properties)
		
	
class PumpkinGhost(Opponent):
	def __init__(self, curr_path, properties):
		Opponent.__init__(self, curr_path, properties)
		self.helmet_properties = properties['helmet']
		self.helmet_displacement = self.helmet_properties.get('displacement', Vector(0, self.size / 3.5))
		self.light_properties = properties['light']
		self.light_colors = self.light_properties['color']
		self.light_displacement = self.light_properties.get('displacement', Vector(0, self.size / 6.0))
		self.light_scale = self.light_properties['scale']
		self.light_image = self.light_properties['image']
		self.helmet = None
		self.helmet_light = None
		
		# Animation
		self.last_light_switch_time = time()
		self.last_frame_time = 0
		self.light_anim_steps = 0
		self.current_anim_step = 0
		self.anim_interval = 2
		self.delta_time = .05
		self.colors_steps = [0, 0, 0] 	# red, green, blue
		self.is_light_on = False
		
		self.__create_helmet()
		self.__calc_light_anim_step()
		
	def update(self):
		Opponent.update(self)
		self.move_helmet()
		self.animate_light()	
		
	def animate_light(self):
		if self.helmet_light:
			curr_time = time()
			if curr_time - self.last_frame_time >= self.delta_time:
				self.last_frame_time = curr_time
				self.current_anim_step += 1
				if self.current_anim_step >= self.light_anim_steps:
					self.current_anim_step = 0
					self.is_light_on = not self.is_light_on
					self.anim_interval = random.randint(10, 30) / 10.0
					self.__calc_light_anim_step()
				self.__change_light_color()
		
	def move_helmet(self):
		self.helmet.position = self.position + self.helmet_displacement
		self.helmet_light.position = self.position + self.light_displacement

	# ----- Used outside this class -----	
	def spawn_opponent(self):
		game.add(self)
		if self.helmet_light:
			game.add(self.helmet_light)
		if self.helmet:
			game.add(self.helmet)
		
	def remove_opponent(self):
		if self.helmet_light:
			game.remove(self.helmet_light)
		if self.helmet:
			game.remove(self.helmet)
		game.remove(self)
		
	# ----- Helpers -----
	def __create_helmet(self):
		self.helmet_properties.update({ 'position': self.position + self.helmet_displacement, 'size': self.size })
		self.helmet = Block(self.helmet_properties)
		self.helmet_light = Block({ 'position': self.position + self.light_displacement, 'scale': self.light_scale, 'image': self.light_image, 'color': self.light_colors['off'], 'size': self.size })
		
	def __calc_light_anim_step(self):
		self.light_anim_steps = self.anim_interval / self.delta_time
		self.colors_steps = [(c2 - c1) / self.light_anim_steps for c2, c1 in zip(self.light_colors['off'], self.light_colors['on'])]
			
	def __change_light_color(self):
		curr_color = self.helmet_light.color
		d_r, d_g, d_b = self.colors_steps
		if self.is_light_on:
			self.helmet_light.color = Color(curr_color.r + d_r, curr_color.g + d_g, curr_color.b + d_b)
		else:
			self.helmet_light.color = Color(curr_color.r - d_r, curr_color.g - d_g, curr_color.b - d_b)


class AlienGhost(PumpkinGhost):
	def __init__(self, curr_path, properties):
		properties['light']['displacement'] = Vector(0, properties['size'] / 7.0)
		properties['helmet']['displacement'] = Vector(0, properties['size'] / 7.0)
		PumpkinGhost.__init__(self, curr_path, properties)
		self.helmet.color = self.color
	
		
#	Health Bar and Point Counter		
###########################################
class Digit():
	def __init__(self, vect, size, dgt_color, bg_color):
		self.digit = Block({'position': vect, 'size': size, 'color': bg_color, 'image': 94})
		self.digit_bg =  Block({'position': vect, 'size': size, 'scale': .9, 'color': dgt_color, 'image': 55})
		
	def load_digit(self):
		game.add(self.digit_bg)
		game.add(self.digit)
		
	def remove_digit(self):
		game.remove(self.digit_bg)
		game.remove(self.digit)
		

class Counter:
	def __init__(self, num_digits, justify, vect, digit_size, digit_gap, digit_color, bg_color):
		self.value = 0
		self.digit_color = digit_color
		self.bg_color = bg_color
		
		self.justify = justify
		self.digit_sprites = []
		self.vect = vect
		self.num_digits = num_digits
		self.digit_size = digit_size
		self.digit_gap = digit_gap
		
		self.digit_props = (self.digit_size, self.digit_color, self.bg_color)
		self.create_counter()
		
	def create_counter(self):
		vect = self.vect
		invariable_props = self.digit_props
		if self.justify == 'left':
			for i in range(self.num_digits):
				self.__add_digit_to_counter(vect + Vector((self.digit_size + self.digit_gap)*i, 0), *invariable_props)
		elif self.justify == 'center':
			one_side_digits_count = self.num_digits // 2
			if one_side_digits_count % 2 == 0:
				vect += Vector(self.digit_size/2.0, 0)
			vect -= Vector(self.digit_size * one_side_digits_count, 0)
			for i in range(self.num_digits):
				self.__add_digit_to_counter(vect + Vector((self.digit_size + self.digit_gap)*i, 0), *invariable_props)
		elif self.justify == 'right':
			for i in range(self.num_digits):
				self.__add_digit_to_counter(vect - Vector((self.digit_size + self.digit_gap)*i, 0), *invariable_props)
		
	def load_counter(self):
		for digit in self.digit_sprites:
			digit.load_digit()
		
	def remove_counter(self):
		for digit in self.digit_sprites:
			digit.remove_digit()
	
	def update_counter(self, value): 
		self.value = abs(int(round(value)))
		curr_num_digits = len(str(self.value))
		delta_dgt_count = curr_num_digits - self.num_digits
		if curr_num_digits > self.num_digits:
			self.num_digits = curr_num_digits 
		if delta_dgt_count > 0:
			self.__add_remain_digit_sprites(delta_dgt_count)
		current_value = str(self.value).zfill(self.num_digits)
		zipped_vals = zip(self.digit_sprites, current_value  if self.justify != 'right' else reversed(current_value))
		for dgt_sprite, dgt_value in zipped_vals:
			dgt_sprite.digit.image = 94 + int(dgt_value)
			
	def __add_digit_to_counter(self, *props):
		digit_sprite = Digit(*props)
		self.digit_sprites.append(digit_sprite)
		return digit_sprite
				
	def __add_remain_digit_sprites(self, delta_dgt_count):
		vect = self.vect
		digit_sprite = None
		# Move all sprites to the left (to align all the digits)
		if self.justify == 'center':
			align_vect = Vector(delta_dgt_count * (self.digit_size + self.digit_gap) * .5, 0)
			for dgt in self.digit_sprites:
				dgt.digit.position -= align_vect
				dgt.digit_bg.position -= align_vect
		for i in range(delta_dgt_count):
			if self.justify == 'left' or self.justify == 'center':
				digit_sprite = self.__add_digit_to_counter(self.digit_sprites[-1].digit.position + Vector(self.digit_size + self.digit_gap, 0), *self.digit_props)
			elif self.justify == 'right':
				digit_sprite = self.__add_digit_to_counter(self.digit_sprites[-1].digit.position - Vector(self.digit_size + self.digit_gap, 0), *self.digit_props)
			digit_sprite.load_digit()
		
			
class Dot(Sprite):
	def __init__(self, vect, size, color):
		self.image = 55
		self.size = size
		self.color = Color(*color)
		self.position = vect
		

class TimeCounter(Sprite):
	def __init__(self, start_time, vect, digit_size, digit_gap, nums_gap, digit_color, bg_color):
		self.size = 0
		game.add(self)
		
		self.vect = vect
		self.nums_gap = nums_gap
		self.digit_size = digit_size
		self.digit_gap = digit_gap
		self.digit_color = digit_color
		self.bg_color = bg_color
		self.start_time = start_time
		self.elapsed_time = 0
		self.current_time = [0, 0, 0] 		# hours, minutes, seconds
		self.counters = [None, None, None] 		# hours, minutes, seconds
		self.dots = [[None, None], [None, None]] 	# between hours and minutes, between minutes and seconds
		self.show_hours = False
		
		self.create_counter()
		
		# Dots animation
		self.last_anim_time = time()
		self.deta_time = 1
		self.are_dots_shown = True
		
	def update(self):
		curr_time = time()
		self.elapsed_time = curr_time - self.start_time
		self.update_time()
		self.update_counter()
		self.animate_dots(curr_time)
			
	def update_time(self):
		elapsed_time = self.elapsed_time
		seconds = int(elapsed_time % 60)
		minutes = int((elapsed_time - seconds) % 3600 / 60)
		hours = int((elapsed_time - (minutes * 60 + seconds)) / 3600)
		self.current_time = [hours, minutes, seconds]
		if hours > 0:
			if not self.show_hours:
				self.counters[0].load_counter()
			self.show_hours = True
			
	def update_counter(self):
		for num, counter in zip(self.current_time, self.counters):
			counter.update_counter(num)
			
	def create_counter(self):
		counter_props = (self.digit_size, self.digit_gap, self.digit_color, self.bg_color)
		self.counters[2] = Counter(2, 'center', self.vect + Vector(self.digit_size + self.nums_gap, 0), *counter_props)
		self.counters[1] = Counter(2, 'center', self.vect - Vector(self.digit_size, 0), *counter_props)
		self.counters[0] = Counter(2, 'center', self.vect - Vector(self.digit_size * 3 + self.nums_gap, 0), *counter_props)
		
		dot_props = (self.digit_size / 5.0, self.digit_color)	
		self.dots[0][0] = Dot(self.vect - Vector(self.digit_size * 3 - self.nums_gap * 1.5, self.digit_size / 7.5), *dot_props)
		self.dots[0][1] = Dot(self.vect - Vector(self.digit_size * 3 - self.nums_gap * 1.5, -self.digit_size / 7.5), *dot_props)
		self.dots[1][0] = Dot(self.vect - Vector(self.digit_size * .5, self.digit_size / 7.5), *dot_props)
		self.dots[1][1] = Dot(self.vect - Vector(self.digit_size * .5, -self.digit_size / 7.5), *dot_props)
		
	def load_counter(self):
		for i in range(len(self.counters)):
			if self.show_hours or i > 0:
				self.counters[i].load_counter()
				
	def remove_counter(self):
		for counter in self.counters:
			if counter:
				counter.remove_counter()
		self.remove_dots()
				
	def load_dots(self):
		for dot in self.dots[1]:
			game.add(dot)
		if self.show_hours:
			for dot in self.dots[0]:
				game.add(dot)
	
	def remove_dots(self):
		for dot in self.dots[1]:
			game.remove(dot)
		if self.show_hours:
			for dot in self.dots[0]:
				game.remove(dot)
				
	def animate_dots(self, curr_time):
		if curr_time - self.last_anim_time >= self.deta_time:
			self.last_anim_time = curr_time
			if self.are_dots_shown:
				self.remove_dots()
			else:
				self.load_dots()
			self.are_dots_shown = not self.are_dots_shown	
				
#	GUI
class Hp:
	def __init__(self, max_hp_value, justify, first_bar_vect, bar_size, bar_colors):
		self.hp_value = 0
		self.first_bar_pos = first_bar_vect
		self.bar_size = bar_size
		self.bar_colors = bar_colors
		self.max_hp_value = max_hp_value
		
		self.hp_bars = {}
		self.hp_bars['bg'] = []
		self.hp_bars['outline'] = []
		
		self.create_hp_bar()
		
	def update_hp(self, hp_value):
		self.hp_value = min(hp_value, self.max_hp_value)
		i = 0
		self.last_frame_time = self.last_anim_time = time()
		for bg in self.hp_bars['bg']:
			if i < self.hp_value:
				bg.color = Color(*self.bar_colors['available'])
			else:
				bg.color = Color(*self.bar_colors['not_available'])
			i += 1
		
	def load_counter(self):
		for bg in self.hp_bars['bg']:
			game.add(bg)
		for outline in self.hp_bars['outline']:
			game.add(outline)
			
	def remove_counter(self):
		for outline in self.hp_bars['outline']:
			game.remove(outline)
		for bg in self.hp_bars['bg']:
			game.remove(bg)
		
	def create_hp_bar(self):
		for i in range(self.max_hp_value):
			self.hp_bars['bg'].append(self.__get_bar_sprite(69, i, self.bar_colors['not_available']))
			self.hp_bars['outline'].append(self.__get_bar_sprite(70, i, self.bar_colors['outline']))
		
	def __get_bar_sprite(self, img, from_right_idx, color):
		properties = { 'image': img, 'color': color, 'size': self.bar_size, 'position': self.first_bar_pos - Vector(self.bar_size * from_right_idx, 0) }
		return Block(properties)
		
		
class Messages:
	def __init__(self, data):
        	# Data useful in messages
		self.lang = 'en'
		self.messages = {}
		self.messages['pl'] = {}
		self.messages['en'] = {}
		self.message_wrapper = '========= PAC-MAN =========\n%s\n========================='
        	add_page = lambda dct, txt: dct.append(txt)
	
		# 	Start, Win & Lose messages
		self.messages['intro'] = '''
			👋 Welcome to the game
			Pac-Man.
			Please select your language before starting the game. Type in:
			> 'en' - English (default)
			> 'pl' - Polish
			If you press 'Enter' without specifying the language, the default one will be loaded.
		'''
		
		# 	ENGLISH language messages
		self.messages['en']['start'] = '''
			⏩ If you encounter any trouble or you don\'t know how to play, please press one of the following keys on your keyboard:
			[ I ] - open a game instruction page
			[ C ] - open controls information page
			Close this window before pressing the button on your keyboard.
		'''
		self.messages['en']['game-over'] = '''
			💀 Game over!
			You are run out of lives.
			Try playing again from the beginning.
			GAINED POINTS: %s
			GAMEPLAY TIME: %s
		'''
		self.messages['en']['game-finished'] = '''
			🏆 Congratulations!
			You have just finished the game.
			There are no more levels available to play.
			GAINED POINTS: %s
			GAMEPLAY TIME: %s
		''' 
		#		Extra messages
		self.messages['en']['continue'] = '\n❌ (Press \'Enter\' to continue.)'
		
		#		Instruction & Controls
	       	self.messages['en']['instruction'] = []
	       	self.messages['en']['pagination'] = "\n=========================\n<= 'p'              Page: %s/%s              'n' =>"
	       	en_instruction_page_1 = '''
			⌨️ CONTROLS:
			To open the controls page, close that window and then press the [C] button.
            		🔑 KEYS:
            		You aim to grab all the keys located in particular segments of the map. In every segment there is only one key to pick up. Keys also have another functionality, because they are used for opening gates, therefore, to enable the player to go to the next segment of the map.
            		🌀 TELEPORT:
            		Having picked up all keys, you will be able to go to the next level. At first, your assignment will be finding the red key randomly spawned on the map. Keep track of the key falling down to notice which way should you go. Then, head to the teleport. Afterward the next level will be loaded.
            		💎 DIAMONDS:
            		Pick up diamonds to raise your point count by +5.
            		⚪ COINS:
			The points gained by picking coins can vary. Depending on the coin size, you will gain either +1 or +2 points.
		'''
	       	en_instruction_page_2 = '''
			🔘 RAGE MODE:
            		The colorful pulsating ball activates the rage mode. During the rage mode, player velocity increases, and the speed of opponents' movement raises likewise. The player becomes immortal for 5 seconds (the red counter adjacent to the hp bar indicates the remaining time). During the rage mode it is profitable to collide with the opponents. This will result in their temporary death (opponents are respawned in {opponent_respawn_time}s). Every opponent kill will be prized by increasing the number of collected points and prolonging the duration of the rage mode by {opponent_kill_bonus_time}s.  
            		The rage mode can be also activated by the player (see controls). The interval of possible activation times starts from {rage_mode_activation_interval}s and decreases by {rage_mode_delta_activation_interval}s every time when the player achieves another threshold of points (on every {activated_rage_next_point_interval} points). This rage mode works similarly to the one, activated by picking up the colorful ball. The next possible activation time is displayed on the counter located above the points counter on the left side of the screen.
	       	'''.format(opponent_respawn_time=data['opponent_respawn_time'], opponent_kill_bonus_time=data['opponent_kill_bonus_time'], rage_mode_activation_interval=data['rage_mode_activation_interval'], rage_mode_delta_activation_interval=data['rage_mode_delta_activation_interval'], activated_rage_next_point_interval=data['activated_rage_next_point_interval'])
	       	en_instruction_page_3 = '''
			❤ HEART POINTS BAR: 
            		The heart points bar shows the remaining heart points of the player. The hp is regenerated in intervals of {hp_increase_interval}s. The number of maximum heart points is being increased by {lvl_up_hp_increase} on loading the next level and overall cannot exceed the number of {max_player_hp} heart points. The remaining time to the hp increase moment is indicated by the counter placed above hp bar (the counter is visible only when the player has lost some lives).
            		💀 DEATH PROTECTION:
            		After the death, the player is immediately respawned in the initial segment of the currently loaded map. The player is protected after being respawned for {player_death_protection_time}s (that means: player becomes immortal for such period). During the period of protection player is changing colors and the counter with a black background, which is placed on the left side of the hp bar, is showing the remaining time.
		'''.format(hp_increase_interval=data['hp_increase_interval'], lvl_up_hp_increase=data['lvl_up_hp_increase'], max_player_hp=data['max_player_hp'], player_death_protection_time=data['player_death_protection_time'])
		en_instruction_page_4 = '''
			👽 OPPONENTS: 
			There are 3 different types of opponents in the game:
			■ DEFAULT GHOST:
			√ slow (speed of {ghost[speed]}),
			√ can go through the gates,
			√ gives {ghost[point_count]} points after being killed,
			■ HALOWEEN GHOST:
			√ fast (speed of {pumpkin_ghost[speed]}),
			√ cannot go through the gates,
			√ gives {pumpkin_ghost[point_count]} points after being killed,
			■ ALIEN GHOST:
			√ very fast (speed of {alien_ghost[speed]}),
			√ cannot go through the gates,
			√ gives {alien_ghost[point_count]} points after being killed,
		'''.format(ghost=data['ghost'], pumpkin_ghost=data['pumpkin-ghost'], alien_ghost=data['alien-ghost'])
	       
		add_page(self.messages['en']['instruction'], en_instruction_page_1)
	      	add_page(self.messages['en']['instruction'], en_instruction_page_2)
	       	add_page(self.messages['en']['instruction'], en_instruction_page_3)
		add_page(self.messages['en']['instruction'], en_instruction_page_4)
	
	
		self.messages['en']['controls'] = '''
			⌨️ CONTROLS:
			Changing the direction of the player movement:
			🡹 / [ W ] - move up
			🡸 / [ A ] - move left
			🡻 / [ S ] - move down
			🡺 / [ D ] - move right
			Activating the rage mode:
			[ SPACE ] - activates the rage mode when possible (see more in the game instruction)
			Others:
			[ I ] - open a game instruction page
			[ C ] - open controls information page
			[ F2 ] - toggle GUI (turns on/off the visibility of counters and the hp bar)
		''' 
		#	Lvl unlock messages
		self.messages['en']['unlock-key-pick'] = '''
			🔓 New level unlocked.
            		Head to the teleport in order to begin the next level.
		'''
		self.messages['en']['unlock-key-spawn'] = '''
			🔑 Congratulations!
            		You have managed to grab all the keys.
            		Now keep track of the unlock key falling down. You have to find it on the map and then head to the teleport. Afterward, there will be the next level loaded.
		'''
		self.messages['en']['loading-next-lvl'] = '''
			⌛ Loading the next map...
			📈 DIFFICULTY LEVEL: %s
			🗺️ MAP NAME: %s
		'''
		self.messages['en']['find-unlock-key'] = '''
			🔑 Please find the unlock key on the map to use the teleport.
		'''
		self.messages['en']['pick-all-keys'] = '''
			🔑 Pick all the keys located in different segments of the map before coming back to the teleport.
            		There is only one key placed in a particular segment.
            		After collecting all the keys, another key will be located in a random segment. Pick it up to unlock the next level.
		'''
		
		# 	POLISH language messages
		self.messages['pl']['start'] = '''
			⏩ Jeżeli napotkasz jakiekolwiek problemy, dotyczące gry lub nie wiesz, jak grać, naciśnij jeden z poniższych klawiszy na klawiaturze:
			[ I ] - otwiera stronę z instrukcją gry
			[ C ] - otwiera stronę ze sterowaniem
			Zamknij to okno przed naciśnięciem przycisków na klawiaturze.
		'''
		self.messages['pl']['game-over'] = '''
			💀 Gra skończona!
			Skończyły Ci się wszystkie życia.
			Spróbuj zagrać ponownie.
			ZDOBYTE PUNKTY: %s
			ŁĄCZNY CZAS GRY: %s
		'''
		self.messages['pl']['game-finished'] = '''
			🏆 Gratulacje!
			Udało Ci się przejść całą grę.
			Nie ma już kolejnych poziomów, na których można grać.
			ZDOBYTE PUNKTY: %s
			ŁĄCZNY CZAS GRY: %s
		''' 
		#		Extra messages
		self.messages['pl']['continue'] = '\n❌ (Naciśnij \'Enter\', aby kontynuować.)'
		
		#		Instruction & Controls
	       	self.messages['pl']['instruction'] = []
	       	self.messages['pl']['pagination'] = "\n=========================\n<= 'p'             Strona: %s/%s             'n' =>"
	       	pl_instruction_page_1 = '''
			⌨️ STEROWANIE:
			Aby otworzyć stronę sterowania, zamknij to okno, a następnie naciśnij klawisz [C].
			🔑 KLUCZE:
			Twoim celem jest zebranie wszystkich kluczy, umieszczonych w poszczególnych segmentach mapy. W każdym z segmentów znajduje się wyłącznie jeden klucz. Klucze mają również dodatkową funkcjonalność, ponieważ pozwalają na otworzenie bramek, a tym samym, umożliwienie graczowi dostania się do następnego segmentu mapy.
			🌀 TELEPORT:
			Po zebraniu wszystkich kluczy, będzie możliwość przejścia na kolejny poziom gry. Aby tego dokonać, najpierw konieczne będzie znalezienie czerwonego klucza, losowo umieszczonego na mapie. Śledź kierunek, w którym spada klucz, aby zauważyć, w którą stronę należy się udać. Po zebraniu tego klucza, udaj się do teleportu, w celu załadowania kolejnego poziomu gry. 
		'''
	       	pl_instruction_page_2 = '''
			💎 DIAMENTY:
			Zbieraj diamenty, by zwiększyć liczbę punktów o +5.
			⚪ MONETY:
			Punkty, zdobyte poprzez zebranie monet, mogą się różnić. W zależności od wielkości monety, możesz otrzymać +1 lub +2 punkty.
			🔘 TRYB FURII:
			Kolorowa, pulsująca kula, aktywuje tryb furii. Podczas tego trybu, zwiększa się prędkość gracza oraz przeciwników. Gracz staje się nieśmiertelny na czas 5 sekund (czerwony licznik, sąsiedni do paska życia, pokazuje pozostały czas). Podczas trybu furii, warto zderzać się z przeciwnikami, w celu pozyskania punktów. W rezultacie, dany przeciwnik zostanie tymczasowo uśmiercony (przeciwnicy są odradzani w czasie {opponent_respawn_time}s). Po zabiciu przeciwnika, gracz otrzyma punkty (liczba punktów zależy od rodzaju przeciwnika), a także przedłużony zostanie czas trwania trybu o {opponent_kill_bonus_time}s.
		'''.format(opponent_respawn_time=data['opponent_respawn_time'], opponent_kill_bonus_time=data['opponent_kill_bonus_time'])
	       	pl_instruction_page_3 = '''
			Tryb furii może również zostać aktywowany przez gracza (patrz strona sterowania). Odstęp pomiędzy kolejnymi momentami, w których gracz może aktywować ten tryb, wynosi początkowo {rage_mode_activation_interval}s i zmniejsza się o {rage_mode_delta_activation_interval}s z osiągnięciem progu kolejnych {activated_rage_next_point_interval} punktów. Tryb aktywowany przez gracza, działa analogicznie do opisanego powyżej. Kolejny możliwy czas aktywacji jest pokazywany przez licznik, ulokowany ponad licznikiem punktów, po lewej stronie ekranu.
			❤ PASEK ZDROWIA: 
			Pasek zdrowia gracza pokazuje pozostałe życia. Życia gracza regenerują się w odstępach czasowych, wynoszących {hp_increase_interval}s. Maksymalna liczba punktów zdrowia jest zwiększana o {lvl_up_hp_increase}, podczas ładowania kolejnego poziomu, a łączna liczba żyć nie może nigdy przekroczyć {max_player_hp}. Pozostały czas do zregenerowania życia jest wskazywany przez licznik, umieszczony nad paskiem zdrowia (widoczny tylko, gdy gracz stracił życie).
		'''.format(rage_mode_activation_interval=data['rage_mode_activation_interval'], rage_mode_delta_activation_interval=data['rage_mode_delta_activation_interval'], activated_rage_next_point_interval=data['activated_rage_next_point_interval'], hp_increase_interval=data['hp_increase_interval'], lvl_up_hp_increase=data['lvl_up_hp_increase'], max_player_hp=data['max_player_hp'])
		pl_instruction_page_4 = '''
			💀 OCHRONA PRZED ŚMIERCIĄ:
			Po śmierci, gracz zostaje natychmiastowo odrodzony w początkowym segmencie mapy. Po odrodzeniu, gracz posiada ochronę przez okres {player_death_protection_time} sekund (tzn. gracz staje się nieśmiertelny przez ten czas). Podczas ochrony, gracz zmienia kolory, a czarny licznik, umieszczony obok paska zdrowia, pokazuje czas do jej zakończenia.
			👽 PRZECIWNICY: 
			W grze występują 3 różne rodzaje przeciwników:
			■ DOMYŚLNY DUCH:
			√ powolny (prędkość wynosi {ghost[speed]}),
			√ może przenikać przez bramki,
			√ daje {ghost[point_count]} punktów po zabiciu,
			■ DUCH HALLOWEENOWY:
			√ szybki (prędkość wynosi {pumpkin_ghost[speed]}),
			√ nie może przechodzić przez bramki,
			√ daje {pumpkin_ghost[point_count]} punktów po zabiciu,
			■ DUCH KOSMITA:
			√ bardzo szybki (prędkość wynosi {alien_ghost[speed]}),
			√ nie może przechodzić przez bramki,
			√ daje {alien_ghost[point_count]} punktów po zabiciu,
		'''.format(player_death_protection_time=data['player_death_protection_time'], ghost=data['ghost'], pumpkin_ghost=data['pumpkin-ghost'], alien_ghost=data['alien-ghost'])
	       
		add_page(self.messages['pl']['instruction'], pl_instruction_page_1)
	      	add_page(self.messages['pl']['instruction'], pl_instruction_page_2)
	       	add_page(self.messages['pl']['instruction'], pl_instruction_page_3)
		add_page(self.messages['pl']['instruction'], pl_instruction_page_4)
	
	
		self.messages['pl']['controls'] = '''
			⌨️ STEROWANIE:
			Zmienianie kierunku poruszania się gracza:
			🡹 / [ W ] - w górę
			🡸 / [ A ] - w lewo
			🡻 / [ S ] - w dół
			🡺 / [ D ] - w prawo
			Aktywowanie trybu furii:
			[ SPACJA ] - aktywuje tryb furii, kiedy jest to możliwe (po więcej informacji zajrzyj do instrukcji)
			Pozostałe:
			[ I ] - otwiera stronę instrukcji gry
			[ C ] - otwiera stronę sterowania
			[ F2 ] - przełącza GUI (włącza lub wyłącza widoczność liczników i paska zdrowia)
		''' 
		#	Lvl unlock messages
		self.messages['pl']['unlock-key-pick'] = '''
			🔓 Nowy poziom został odblokowany.
			Udaj się do teleportu, aby załadować kolejną mapę.
		'''
		self.messages['pl']['unlock-key-spawn'] = '''
			🔑 Gratulacje!
			Udało Ci się zebrać wszystkie klucze, jakie znajdowały się na mapie.
			Teraz śledź kierunek, w którym będzie spadał czerwony klucz, co pozwoli Tobie go łatwiej znaleźć. Po odnalezieniu klucza, możesz się udać do teleportu, w celu załadowania następnego poziomu gry.
		'''
		self.messages['pl']['loading-next-lvl'] = '''
			⌛ Ładowanie mapy...
			📈 POZIOM TRUDNOŚCI: %s
			🗺️ NAZWA MAPY: %s
		'''
		self.messages['pl']['find-unlock-key'] = '''
			🔑 Odnajdź czerwony klucz na mapie, aby odblokować teleport.
		'''
		self.messages['pl']['pick-all-keys'] = '''
			🔑 Zbierz wszystkie klucze, umieszczone w różnych segmentach mapy, zanim wrócisz do teleportu.
			W każdym segmencie znajduje się wyłącznie jeden klucz.
			Po tym, jak zbierzesz wszystkie klucze, kolejny klucz zostanie umieszczony w losowym segmencie. Podnieś ten klucz, by odblokować kolejny poziom.
		'''
		
	def message(self, txt, use_message_wrapper=True, extra_text='', params=()):
		if 'pix.breakpoint' in txt:
			txt = '\n'.join(self.clear_string(txt))
		out_text = txt + extra_text
		if use_message_wrapper:
			out_text = self.message_wrapper % out_text
		if params:
			out_text = out_text % params
		return message.show(out_text)
		
	def open_instruction_page(self):
		pages = self.messages[self.lang]['instruction']
		curr_page_idx = 0
		page_count = len(pages)
		input_val = '_'
		while input_val != '':
			input_val = self.message(pages[curr_page_idx], extra_text=self.messages[self.lang]['pagination'], params=(curr_page_idx+1, page_count)).lower()
			if input_val == 'n' and curr_page_idx + 1 < page_count:
				curr_page_idx += 1
			elif input_val == 'p' and curr_page_idx - 1 >= 0:
				curr_page_idx -= 1


# 	Game Class		
#######################
class Game(Messages, Board, Sprite):
	def __init__(self):
		Board.__init__(self)
		self.player = None
		self.size = 0
		game.add(self)
		
		# Player hp state
		self.player_hp = 3
		self.current_max_player_hp = 3
		self.max_player_hp = 6
		self.hp_heart_size = 8
		self.lvl_up_hp_increase = 2
		self.player_death_protection_time = 3
		self.player_protection_start_time = 0
			
		# Rage mode
		self.rage_mode = False
		self.rage_speed = 0
		self.rage_mode_duration = 0
		self.rage_mode_start_time = 0
		self.rage_mode_end_time = 0
		self.opponent_respawn_time = 1
		self.opponent_kill_bonus_time = 1
		self.min_opponent_respawn_range = 50
		self.max_opponent_respawn_search_repetitions = 10
		
		# Counters
		self.temporary_counters_vect = Vector(90 - self.hp_heart_size * self.current_max_player_hp, -92)
		self.game_start_time = 0
		self.play_time_counter_properties = (Vector(78, 92), 10, -2, 2, (0,0,0), (255, 255, 255))
		self.points_counter_properties = (4, 'left', Vector(-92, -92), 10, -2, (0,0,0), (255, 255, 255))
		self.next_rage_counter_properties = (1, 'left', Vector(-92, -82), 10, -2, (0, 175, 81), (255,255,255))
		self.rage_mode_counter_properties = (1, 'right', self.temporary_counters_vect, 10, -2, (255,255,255), (255, 13, 0))
		self.next_hp_increase_counter_properties = (1, 'right', Vector(92, -82), 10, -2, (255, 7, 0), (255,255,255))
		self.protection_time_counter_properties = (1, 'right', self.temporary_counters_vect, 10, -2, (255,255,255), (0,0,0))
		self.hp_bar_properties = ('right', Vector(92, -92), self.hp_heart_size, {'outline': (0,0,0), 'available': (255, 0, 0), 'not_available': (137, 137, 137)})
		
		self.hp_bar = None
		self.points_counter = None
		self.rage_mode_counter = None
		self.next_rage_counter = None
		self.play_time_counter = None
		self.protection_time_counter = None
		self.next_hp_increase_counter = None
		
		# Game state
		self.gained_points = 0
		self.game_start_time = 0
		self.last_hp_increase_time = 0
		self.last_rage_mode_time = 0 
		self.hp_increase_interval = 180
		self.rage_mode_activation_interval = 60
		self.activated_rage_duration = 5
		self.activated_rage_speed = 3.5 * SPEED_MULTIPLIER
		self.activated_rage_next_point_threshold = 1000
		self.activated_rage_next_point_interval = 1000
		self.rage_mode_min_activation_interval = 3
		self.rage_mode_delta_activation_interval = 5 # How much interval decreases when new point level achieved
		
		self.animated_unlock_key = None
		self.is_unlock_key_spawned = False
		self.are_moving_sprites_reloaded = True
		
		# GUI
		self.is_gui_visible = True
		self.button_press_start_time = 0
		self.button_press_delay = .5
		self.game_keys_dict = {}
		self.game_keys_dict['f2'] = lambda: self.__toogle_gui_visibility()
		self.game_keys_dict['i'] = lambda: self.open_instruction_page()
		self.game_keys_dict['c'] = lambda: self.message(self.messages[self.lang]['controls'], extra_text=self.messages[self.lang]['continue'])
		
		# Messages
		messages_data = {}
		messages_data['opponent_respawn_time'] = self.opponent_respawn_time
		messages_data['opponent_kill_bonus_time'] = self.opponent_kill_bonus_time
		messages_data['rage_mode_activation_interval'] = self.rage_mode_activation_interval
		messages_data['rage_mode_delta_activation_interval'] = self.rage_mode_delta_activation_interval
		messages_data['activated_rage_next_point_interval'] = self.activated_rage_next_point_interval
		messages_data['hp_increase_interval'] = self.hp_increase_interval
		messages_data['lvl_up_hp_increase'] = self.lvl_up_hp_increase
		messages_data['max_player_hp'] = self.max_player_hp
		messages_data['player_death_protection_time'] = self.player_death_protection_time
		messages_data['ghost'] = self.blocks['ghost']
		messages_data['pumpkin-ghost'] = self.blocks['pumpkin-ghost']
		messages_data['alien-ghost'] = self.blocks['alien-ghost']
		Messages.__init__(self, messages_data)
		
		self.last_teleport_collision_time = 0
		self.teleport_collision_message_update_duration = 3
		
	def update(self):
		if self.player:
			self.handle_gui()
			self.trace_player_position()
			self.handle_player_collisions() # E.g. with opponents, coin, key, diamond etc.
			self.handle_rage_mode()
			self.handle_hp_increase()
			self.handle_opponents_respawn()
			self.update_protection_time_counter()
			self.update_next_rage_mode()
		
	# ---------- Initialisation and game state ----------
	def start(self):
        	sound.play(sounds['gameplay-music'])
		self.__play_intro()
		self.message(self.messages[self.lang]['start'], extra_text=self.messages[self.lang]['continue'])
		self.game_start_time = self.last_rage_mode_time = self.current_counter_time = time()
		self.load_lvl(1)
		
	def game_over(self):
        	sound.play(sounds['game-over'])
		self.__clear_whole_map()
		self.__play_game_over()
		
	def game_finished(self):
            	sound.play(sounds['game-finished'])
		self.__clear_whole_map()
		self.__play_game_finished()
		
	def load_lvl(self, difficulty_lvl_num):
		self.next_lvl_unlocked = False
		self.lvl_unlock_key = None
		self.init_level(difficulty_lvl_num)
		self.load_start_segment()
		
	def load_start_segment(self):
		self.current_segment_id = self.boards[self.difficulty_levels[self.current_difficulty_level]]['start_segment']['segment_id']
		self.load_map()
		self.spawn_opponents()
		self.spawn_player()
		self.spawn_moving_sprites()
		self.load_counters()
		
	def stop_opponents_movement(self):
		for opponent in self.current_opponents:
			opponent.stop_moving()
			
	def start_opponents_movement(self):
		for opponent in self.current_opponents:
			opponent.start_moving()
		
	# ---------- Map reload and spawn Sprites ----------
	def spawn_player(self, is_stopped=False):
		diff_lvl = self.difficulty_levels[self.current_difficulty_level]
		start_segment = self.boards[diff_lvl]['start_segment']
		self.player = Player(start_segment['spawn_coords'], self.current_path, 0, self.block_size)
		self.player.is_stopped = is_stopped
		self.update_player_map_bounds()
		self.update_player_obstacles()
		game.add(self.player)
        	if not self.player.is_stopped:
        		sound.play(sounds['player']['spawn'])
	
	def reload_to_segment(self, segment_id):
		self.remove_counters()
		Board.reload_to_segment(self, segment_id)
		self.respawn_player()
		self.update_opponents_positions()
        	self.spawn_opponents()
		self.update_player_map_bounds()
		self.update_player_obstacles()
		self.spawn_moving_sprites()
		self.update_rage_mode()
		self.load_counters()
		if self.animated_unlock_key:
			game.remove(self.animated_unlock_key)
			self.animated_unlock_key = None

	def update_player_obstacles(self):
		obstacles = self.get_gate_blocks()
        	self.player.gates_coords = [gate.coords for gate in self.current_gates]
		self.player.update_obstacles(obstacles)
		
	def update_player_map_bounds(self):
		self.player.update_map_bounds(self.block_count_x, self.block_count_y)

	def update_opponents_positions(self):
		for opponent in self.current_opponents:
			coords = self.current_path.get_available_opponent_coords(opponent.current_coords, opponent.next_coords)
			if coords != True:
				opponent.current_coords = coords[0]
				opponent.next_coords = coords[1]
				opponent.position = self.current_path.node_vectors_dict[coords[0]]

	# ---------- Used in .update() method and for reload ----------
	def handle_gui(self):
		curr_time = time()
		if curr_time - self.button_press_start_time >= self.button_press_delay:
			for key, func in self.game_keys_dict.items():
				if game.key(key):
					self.button_press_start_time = curr_time
					func()
	
	def update_next_rage_mode(self):
		curr_time = time()
		remain_time = int(self.rage_mode_activation_interval - (curr_time - self.last_rage_mode_time))
		if remain_time >= 0:
			if self.next_rage_counter:
				self.next_rage_counter.update_counter(int(remain_time+.5))
		elif not self.rage_mode:
			if self.next_rage_counter:
				self.next_rage_counter.remove_counter()
				self.next_rage_counter = None
			if game.key('space'):
				self.last_rage_mode_time = curr_time 
				self.__init_rage_mode(None, self.activated_rage_speed, self.activated_rage_duration)
		
	def update_protection_time_counter(self):
		if self.player:		
			remain_time = int(self.player_death_protection_time - (time() - self.player_protection_start_time) + 1)
			if self.player.death_protection and self.protection_time_counter and remain_time >= 0:
				self.protection_time_counter.update_counter(remain_time)
			elif self.protection_time_counter:
				self.protection_time_counter.remove_counter()
				self.protection_time_counter = None
			
	def trace_player_position(self):
		vect = self.player.position
		if max(abs(vect.x), abs(vect.y)) >= 99:
			next_segment = self.__get_next_segment_id(vect)
			self.reload_to_segment(next_segment)	
			
	def handle_hp_increase(self):
		if self.player_hp < self.current_max_player_hp:
			if not self.next_hp_increase_counter and self.is_gui_visible and self.player:
				self.next_hp_increase_counter = Counter(*self.next_hp_increase_counter_properties)
				self.next_hp_increase_counter.load_counter()
			curr_time = time()
			remain_time = int(self.hp_increase_interval - (curr_time - self.last_hp_increase_time))
			if remain_time < 0:
				self.player_hp = self.player_hp + 1 if self.player_hp + 1 <= self.current_max_player_hp else self.curr_max_player_hp 
				if self.hp_bar:
					self.hp_bar.update_hp(self.player_hp)
				self.last_hp_increase_time = curr_time
			elif self.next_hp_increase_counter:
				self.next_hp_increase_counter.update_counter(remain_time)
		elif self.next_hp_increase_counter:
			self.last_hp_increase_time = time()
			self.next_hp_increase_counter.remove_counter()
			self.next_hp_increase_counter = None
			
	def handle_player_collisions(self):
		if self.current_key:
			if self.player.collide(self.current_key):
				self.__open_segment_gates()
				self.__trace_next_lvl_unlock()
		if self.current_rage and not self.rage_mode:
			if self.player.collide(self.current_rage):
				self.__init_rage_mode(self.current_rage)
		if self.current_points['diamonds']:
			self.__handle_point_collision('diamonds')
		if self.current_points['coins']:
			self.__handle_point_collision('coins')
		if self.current_opponents:
			self.__handle_opponent_collision()
		if self.lvl_unlock_key:
			self.__handle_lvl_unlock_key_collision()
		if self.current_teleport:
			self.__handle_teleport_collision()
			
	def handle_rage_mode(self):
		if self.rage_mode:
			remain_rage_time = self.rage_mode_start_time - time() + self.rage_mode_duration
			if remain_rage_time <= 0:
				self.__stop_rage_mode()
			elif self.rage_mode_counter:
				self.rage_mode_counter.update_counter(remain_rage_time+.5)
			 
	def handle_opponents_respawn(self):
		if self.opponents_respawn_queue.length > 0:
			first_death_time = self.opponents_respawn_queue.peek()['death_time']
			if time() - first_death_time >= self.opponent_respawn_time:
				opponent = self.opponents_respawn_queue.dequeue()['opponent']
				Cls = opponent.__class__.__name__
				name = opponent.name
				coords = self.player.current_coords
				while coords == self.player.current_coords:
					coords = self.__get_opponent_respawn_coords(opponent.next_coords)
				new_opponent = self.create_opponent(eval(Cls), name, coords, False)
				new_opponent.rage_mode = self.rage_mode
				new_opponent.rage_speed = self.rage_speed
				new_opponent.spawn_opponent()
				self.are_moving_sprites_reloaded = False
				#self.load_counters()
		elif not self.are_moving_sprites_reloaded and not self.rage_mode:
			# Reload moving sprites (trees, flames) and counter one second later (to prevent slowing down the game)
			if time() - self.rage_mode_end_time > 1:
				self.are_moving_sprites_reloaded = True
				self.respawn_moving_sprites()
				self.load_counters()
	
	def update_rage_mode(self):
		if self.player:
			self.player.rage_mode = self.rage_mode
			self.player.rage_speed = self.rage_speed
			for opponent in self.current_opponents:
				opponent.rage_mode = self.rage_mode
				opponent.rage_speed = self.rage_speed

	def respawn_player(self):
        	#message.show('prev_coords: {0}\ncurr_coords: {1}\nnext_coords: {2}'.format(self.player.prev_coords, self.player.current_coords, self.player.next_coords))
		game.remove(self.player)
		prev_coords = self.player.current_coords
		player_x, player_y = prev_coords	
        	center_x, center_y = [(n-1)/2.0 for n in self.block_count_x, self.block_count_y]
        	dist_x, dist_y = player_x - center_x, player_y - center_y

        	# Check if player was farther horizontally from the centre than vertically
		if abs(dist_x) > abs(dist_y):
            		player_x = 0 if dist_x > 0 else self.block_count_x-1
       		else:
            		player_y = 0 if dist_y > 0 else self.block_count_y-1
		
		death_protection = self.player.death_protection
		coords = (player_x, player_y)
		self.player = Player(coords, self.current_path, self.player.angle, self.block_size)
		self.player.death_protection = death_protection 
		game.add(self.player)
        	#message.show('Player coords: {0} -> {1}\nMap bounds: x={2}, y={3}'.format(prev_coords, coords, self.block_count_x, self.block_count_y))
            	#message.show('prev_coords: {0}\ncurr_coords: {1}\nnext_coords: {2}'.format(self.player.prev_coords, self.player.current_coords, self.player.next_coords))
		
	def respawn_opponent(self, opponent):
		opponent.remove_opponent()
		self.current_opponents.remove(opponent)
		self.opponents_respawn_queue.enqueue({ 'death_time': time(), 'opponent': opponent })
		
	def remove_counters(self):
		counters = [self.hp_bar, self.points_counter, self.play_time_counter, self.rage_mode_counter, self.protection_time_counter, self.next_rage_counter, self.next_hp_increase_counter]
		if self.play_time_counter:
			game.remove(self.play_time_counter)
		for counter in counters:
			if counter:
				counter.remove_counter()
		self.hp_bar = self.play_time_counter = self.points_counter = self.rage_mode_counter = self.next_rage_counter = self.next_hp_increase_counter = self.protection_time_counter = None
				
	def load_counters(self):
		# Create counters
		self.remove_counters()
		if self.is_gui_visible:
			self.hp_bar = Hp(self.current_max_player_hp, *self.hp_bar_properties)
			self.hp_bar.update_hp(self.player_hp)
			self.points_counter = Counter(*self.points_counter_properties)
			self.points_counter.update_counter(self.gained_points)
			self.play_time_counter = TimeCounter(self.game_start_time, *self.play_time_counter_properties)
			if self.rage_mode:
				self.rage_mode_counter = Counter(*self.rage_mode_counter_properties)
			elif self.player: 
				self.next_rage_counter = Counter(*self.next_rage_counter_properties)
				if self.player.death_protection:
					self.protection_time_counter = Counter(*self.protection_time_counter_properties)
			# Load counters
			counters = [self.hp_bar, self.points_counter, self.play_time_counter, self.rage_mode_counter, self.protection_time_counter, self.next_rage_counter]
			for counter in counters:
				if counter:
					counter.load_counter()
		
	# ---------- Getters ----------
	def __get_opponent_respawn_coords(self, coords):
		for i in range(self.max_opponent_respawn_search_repetitions):
			connected_nodes = self.current_path.node_connections_dict[coords]
			coords = list(connected_nodes)[random.randint(0, len(connected_nodes)-1)]
			coords_vect = self.current_path.node_vectors_dict[coords]
			if (self.player.position - coords_vect).length >= self.min_opponent_respawn_range and i > 2:
				break
		return coords
			
	def __get_current_segment_idx(self, diff_lvl, segment_arr):
		segment_id = self.current_segment_id
		i = j = 0
		for i in range(len(segment_arr)):
			if segment_id in segment_arr[i]:
				j = segment_arr[i].index(segment_id)
				break
		return (i, j)
			
	def __get_next_segment_id(self, player_vect):
		diff_lvl = self.difficulty_levels[self.current_difficulty_level]
		segment_arr = self.boards[diff_lvl]['segments_pattern']
		
		i, j = self.__get_current_segment_idx(diff_lvl, segment_arr)
		if player_vect.x <= -99:
			j-=1
		elif player_vect.x >= 99:
			j+=1
		elif player_vect.y <= -99:
			i+=1
		elif player_vect.y >= 99:
			i-=1
		return segment_arr[i][j]
		
	def __get_time_string(self, time_in_seconds):
		seconds = int(time_in_seconds % 60)
		minutes = int((time_in_seconds - seconds) % 3600 / 60)
		hours = int((time_in_seconds - (minutes * 60 + seconds)) / 3600)
		return '%02d : %02d : %02d' % (hours, minutes, seconds)
		
	# ---------- Helpers ----------
	def __clear_whole_map(self):
        	self.game_end_time = self.play_time_counter.last_anim_time if self.play_time_counter else time()
		self.remove_counters()
		game.remove(self.player)
		self.player = None
		self.clear_map()
		
	def __play_intro(self):
		self.init_level('intro')
		self.load_map()
		self.spawn_player(is_stopped=True)
		self.player.size *= 8
		self.spawn_moving_sprites()
		self.spawn_opponents()
		self.__get_game_language()
		game.remove(self.player)
		self.player = None
		self.clear_map()
		
	def __get_game_language(self):
		lang = ''
		while lang not in self.messages.keys():
			lang = self.message(self.messages['intro'], extra_text=self.messages[self.lang]['continue']).lower()
			if lang == '':
				return
		self.lang = lang	
		
	def __play_game_over(self):
		self.is_gui_visible = False
		self.init_level('lose')
		self.load_map()
		props = { 'scale': 4, 'position': Vector(0, -15), 'angle': 35, 'size': self.block_size }
		bone1 = self.get_block('bone', props)
		props.update({ 'angle': -35 })
		bone2 = self.get_block('bone', props)
		self.add_block_to_game(bone1)
		self.add_block_to_game(bone2)
		props = { 'scale': 4, 'position': Vector(0, 10), 'angle': 0, 'size': self.block_size }
		skull = self.get_block('skull', props)
		self.add_block_to_game(skull)
		self.spawn_opponents()
		self.spawn_moving_sprites()
		self.stop_opponents_movement()
		self.message(self.messages[self.lang]['game-over'], params=(self.gained_points, self.__get_time_string(time() - self.game_start_time)))
		
	def __play_game_finished(self):
		self.is_gui_visible = False
		self.init_level('win')
		self.load_map()
		self.spawn_opponents()
		self.spawn_moving_sprites()
		self.spawn_player()
		self.player.size *= 6
		self.player.stop_moving()
		self.player = None		
		self.message(self.messages[self.lang]['game-finished'], params=(self.gained_points, self.__get_time_string(self.game_end_time - self.game_start_time)))
		
	def __toogle_gui_visibility(self):
		self.is_gui_visible = not self.is_gui_visible 
		self.load_counters()
		
	# Game state modifying methods
	def __open_segment_gates(self):
		game.remove(self.current_key)
		self.current_key = None
		self.open_segment_gates()
		
	def __trace_next_lvl_unlock(self):
		self.remain_keys_num -= 1
		if self.remain_keys_num == 0 and not self.lvl_unlock_key:
			self.__spawn_lvl_unlock_key()
			
	def __spawn_lvl_unlock_key(self):
		cached_segments = self.cached_blocks
		segments_ids = cached_segments.keys()
		if self.current_segment_id in segments_ids:
			segments_ids.remove(self.current_segment_id)
		spawn_segment_id = segments_ids[random.randint(0, len(segments_ids)-1)]
		cached = self.cached_blocks[spawn_segment_id]
		cached_path = cached['path'].node_vectors_dict
		spawn_vect = cached_path.values()[random.randint(0, len(cached_path)-1)]
		partial_vectors = self.__get_anim_key_partial_vectors(self.current_segment_id, spawn_segment_id)
		props = { 'color': (255, 33, 49), 'scale': .8, 'size': self.block_size, 'position': spawn_vect }
		self.lvl_unlock_key = self.get_block('key', props)
		cached['unlock-key'] = self.lvl_unlock_key
		self.animated_unlock_key = AnimatedKey(self.lvl_unlock_key.properties, partial_vectors)
		self.is_unlock_key_spawned = True
		self.message(self.messages[self.lang]['unlock-key-spawn'], extra_text=self.messages[self.lang]['continue'])
		
	def __get_anim_key_partial_vectors(self, curr_seg_id, spawn_seg_id):
		pattern = self.boards[self.difficulty_levels[self.current_difficulty_level]]['segments_pattern']
		curr_coords = spawn_coords = (0,0)
		for i in range(len(pattern)):
			for j in range(len(pattern[i])):
				if pattern[i][j] == curr_seg_id:
					curr_coords = (j, i)
				elif pattern[i][j] == spawn_seg_id:
					spawn_coords = (j, i)
		d_x = spawn_coords[0] - curr_coords[0]
		d_y = curr_coords[1] - spawn_coords[1] # We have to reverse axis (because y axis is increasing while going down)
		return (d_x, d_y)
		
	def __handle_lvl_unlock_key_collision(self):
		if self.player.collide(self.lvl_unlock_key):
			game.remove(self.lvl_unlock_key)
			self.lvl_unlock_key = None
			self.next_lvl_unlocked = True
			cached = self.cached_blocks[self.current_segment_id]
			cached['unlock-key'] = None
			self.message(self.messages[self.lang]['unlock-key-pick'], extra_text=self.messages[self.lang]['continue'])
		
	def __handle_teleport_collision(self):
		curr_time = time()
		if self.player and curr_time - self.last_teleport_collision_time >= self.teleport_collision_message_update_duration:
			if self.player.collide(self.current_teleport):
				if self.next_lvl_unlocked:
					next_lvl_num = self.current_difficulty_level + 1
                    			lvl_name = self.difficulty_levels.get(next_lvl_num, 'win')
                    			map_name = self.boards[lvl_name]['name']
					self.message(self.messages[self.lang]['loading-next-lvl'], extra_text=self.messages[self.lang]['continue'], params=(lvl_name.title() if lvl_name != 'win' else '-', map_name.title()))
                    			sound.play(sounds['teleport'])
					self.__load_next_lvl()
				elif self.is_unlock_key_spawned:
					self.message(self.messages[self.lang]['find-unlock-key'], extra_text=self.messages[self.lang]['continue'])
				else:
					self.message(self.messages[self.lang]['pick-all-keys'], extra_text=self.messages[self.lang]['continue'])
				self.last_teleport_collision_time = curr_time
		
	def __handle_point_collision(self, point_type):
		if self.player.collide(self.current_points[point_type]):
			for pt in self.current_points[point_type]:
				if self.player.collide(pt):
                    			sound.play(sounds['points'][point_type])
					self.current_points[point_type].remove(pt)
					game.remove(pt)
					self.gained_points += pt.point_count
					self.__update_next_rage_mode_interval()
					if self.points_counter:
						self.points_counter.update_counter(self.gained_points)
					
	def __handle_opponent_collision(self):
		if self.player.collide(self.current_opponents):
			for opponent in self.current_opponents:
				if self.player.collide(opponent):
					# Check if player has not finished rage mode (if were initialised)
					if self.rage_mode or self.player.speed > self.player.default_speed:
                        			sound.play(sounds['rage']['opponent-kill'])
						# Respawn opponent if player in the rage mode
						self.respawn_opponent(opponent)
						self.gained_points += opponent.point_count
						self.rage_mode_start_time += self.opponent_kill_bonus_time # Prolong rage execution time when an opponent was killed
						self.last_rage_mode_time += self.opponent_kill_bonus_time
						if self.points_counter:
							self.points_counter.update_counter(self.gained_points)
					elif not self.player.death_protection:
						self.__handle_player_death()
			
	def __handle_player_death(self):
		if self.player_hp - 1 <= 0:
			self.game_over()
		else:
            		sound.play(sounds['player']['death'])
			curr_time = time()
			game.remove(self.player)
			curr_board = self.boards[self.difficulty_levels[self.current_difficulty_level]]
			start_segment = curr_board['start_segment']
			start_segment_id = start_segment['segment_id']
			death_segment_id = self.current_segment_id
			self.player_hp -= 1
			self.last_rage_mode_time = curr_time
			if death_segment_id == start_segment_id:
				self.remove_counters()
				self.spawn_player()
				self.respawn_moving_sprites()
				self.load_counters()
			else:
				if self.animated_unlock_key:
					game.remove(self.animated_unlock_key)
					self.animated_unlock_key = None
				self.clear_map()
				self.load_start_segment()
			if self.is_gui_visible and not self.protection_time_counter:
				self.protection_time_counter = Counter(*self.protection_time_counter_properties)
				self.protection_time_counter.load_counter()
			if self.hp_bar:
				self.hp_bar.update_hp(self.player_hp)
			self.player.death_protection_time = self.player_death_protection_time	
			self.player.death_protection = True
			self.rage_mode = False
			self.update_rage_mode()
			if self.last_hp_increase_time == 0:
				self.last_hp_increase_time = curr_time
			self.player_protection_start_time = curr_time
				
	def __update_next_rage_mode_interval(self):
		if self.gained_points >= self.activated_rage_next_point_threshold:
			self.activated_rage_next_point_threshold += self.activated_rage_next_point_interval
			if self.rage_mode_activation_interval - self.rage_mode_delta_activation_interval >= self.rage_mode_min_activation_interval:
				self.rage_mode_activation_interval -= self.rage_mode_delta_activation_interval 
				
	def __init_rage_mode(self, rage_sprite, rage_speed=0, rage_mode_duration=0):
        	sound.play(sounds['rage']['activation'])
		if rage_sprite:
			game.remove(rage_sprite)
			self.current_rage = None
		self.rage_mode_start_time = time()
		self.rage_mode = True
		self.rage_speed = rage_sprite.speed_boost * SPEED_MULTIPLIER if rage_sprite else rage_speed
		self.rage_mode_duration = rage_sprite.duration if rage_sprite else rage_mode_duration
		self.update_rage_mode()
		if self.is_gui_visible:
			self.rage_mode_counter = Counter(*self.rage_mode_counter_properties)
			self.rage_mode_counter.load_counter()
		self.last_rage_mode_time += self.rage_mode_duration 
		if self.next_rage_counter:
			self.next_rage_counter.remove_counter()
			self.next_rage_counter = None
		
	def __stop_rage_mode(self):
		self.rage_mode = False
		self.update_rage_mode()
		if self.rage_mode_counter:
			self.rage_mode_counter.remove_counter()
		self.rage_mode_counter = None
		if self.is_gui_visible:
			self.next_rage_counter = Counter(*self.next_rage_counter_properties)
			self.next_rage_counter.load_counter()
		self.rage_mode_end_time = time()
		
	def __load_next_lvl(self):
		self.is_unlock_key_spawned = False
		if self.animated_unlock_key:
			game.remove(self.animated_unlock_key)
			self.animated_unlock_key = None
		self.clear_map()
		game.remove(self.player)
		self.current_difficulty_level += 1
		self.current_max_player_hp = self.current_max_player_hp + self.lvl_up_hp_increase if self.current_max_player_hp + self.lvl_up_hp_increase < self.max_player_hp else self.max_player_hp 
		self.player_hp = self.player_hp + self.lvl_up_hp_increase if self.player_hp + self.lvl_up_hp_increase < self.max_player_hp else self.current_max_player_hp
		self.temporary_counters_vect.x = 90 - self.hp_heart_size * self.current_max_player_hp
		if self.current_difficulty_level in self.difficulty_levels:
			self.load_lvl(self.current_difficulty_level)
		else:
			self.game_finished()
		
		
# 	Game Initialisation	
################################
my_game = Game()

# 	INTRO segment
intro_segment = '''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxbbbbbxxxxbbbxxxxxbbbxxxxxxx
xxxxxbbbbbbxxbbbbbxxxbbbbbxxxxxx
xxxxxbbxxbbxbbbxbbbxbbbxbbbxxxxx
xxxxxbbxxbbxbbxxxbbxbbxxxbbxxxxx
xxxxxbbbbbbxbbxxxbbxbbxxxxxxxxxx
xxxxxbbbbbxxbbbbbbbxbbxxxxxxxxxx
xxxxxbbxxxxxbbbbbbbxbbxxxbbxxxxx
xxxxxbbxxxxxbbxxxbbxbbbxbbbxxxxx
xxxxxbbxxxxxbbxxxbbxxbbbbbxxxxxx
xxxxxbbxxxxxbbxxxbbxxxbbbxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxx3xxxxx2xxxxx1xxxxxx&xxxxdxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxbxxxxxbxxxbbbxxxbxxxxbbxxxx
xxxxxbbxxxbbxxbbbbbxxbbxxxbbxxxx
xxxxxbbbxbbbxbbbxbbbxbbbxxbbxxxx
xxxxxbbbbbbbxbbxxxbbxbbbbxbbxxxx
xxxxxbbbbbbbxbbxxxbbxbbbbbbbxxxx
xxxxxbbxbxbbxbbbbbbbxbbbbbbbxxxx
xxxxxbbxxxbbxbbbbbbbxbbxbbbbxxxx
xxxxxbbxxxbbxbbxxxbbxbbxxbbbxxxx
xxxxxbbxxxbbxbbxxxbbxbbxxxbbxxxx
xxxxxbbxxxbbxbbxxxbbxbbxxxxbxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
my_game.set_map_pattern('intro', 'Intro', '1', '1', 32, 32)
my_game.add_map_segment('intro', '1', intro_segment)
my_game.set_map_background('intro', (114, 94, 0))
my_game.set_map_styles('intro', 'block', { 'color': (255, 234, 135) })
my_game.set_map_styles('intro', 'diamond', { 'scale': 3 })
my_game.set_map_styles('intro', 'ghost', { 'scale': 5 })
my_game.set_map_styles('intro', 'pumpkin-ghost', { 'scale': 5 })
my_game.set_map_styles('intro', 'alien-ghost', { 'scale': 5 })

# 	WIN segment
win_segment = '''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxssxxsoxxxosoxxxooxxxosxxxxx
xxxxxssxxssxxsoosoxxosxxxosxxxxx
xxxxxosxxosxoosxosoxosxxxsoxxxxx
xxxxxosssooxosxxxssxsoxxxssxxxxx
xxxxxxsssoxxsoxxxsoxooxxxooxxxxx
xxxxxxxsoxxxosxxxssxooxxxssxxxxx
xxxxxxxsoxxxooxxxsoxssxxxooxxxxx
xxxxxxxosxxxossxoosxsosxsooxxxxx
xxxxxxxssxxxxssosoxxxsosssxxxxxx
xxxxxxxosxxxxxoosxxxxxosoxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxx1xxx2xxx3xxxx
xxxxx&xxxxxkxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxfxfxfxfxfxxxx
xxxxxxxxxxxxxxxxxxfxfxfxfxfxfxxx
xxxxxxxxxxxxxxxxxxxfxfxfxfxfxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxooxxxsoxxxosoxxxsxxxxosxxxx
xxxxxsoxxxsoxxossosxxsoxxxsoxxxx
xxxxxooxxxooxosoxssoxoooxxsoxxxx
xxxxxsoxxxooxooxxxsoxsosoxsoxxxx
xxxxxosxoxosxooxxxsoxssooossxxxx
xxxxxossoosoxsoxxxosxsossossxxxx
xxxxxoosooooxsoxxxosxssxsoooxxxx
xxxxxoooxsooxoosxosoxooxxsosxxxx
xxxxxsoxxxooxxooosoxxooxxxssxxxx
xxxxxoxxxxxsxxxsooxxxssxxxxoxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

'''
my_game.set_map_pattern('win', 'Win', '1', '1', 32, 32)
my_game.add_map_segment('win', '1', win_segment)
my_game.set_map_background('win', (85, 33, 0))
my_game.set_map_styles('win', 'flame', { 'scale': (1.5, 1.8) })
my_game.set_map_styles('win', 'spruce', { 'scale': (1.2, 1.8) })
my_game.set_map_styles('win', 'oak', { 'scale': (1.1, 1.5) })
my_game.set_map_styles('win', 'ghost', { 'scale': 4, 'angle': 15 })
my_game.set_map_styles('win', 'pumpkin-ghost', { 'scale': 4 })
my_game.set_map_styles('win', 'alien-ghost', { 'scale': 4, 'angle': -15 })
my_game.set_map_styles('win', 'key', { 'scale': 4, 'angle': -25, 'color': (255, 0, 0) })

# 	GAME OVER segment
lose_segment = '''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxsssxxxxxsssxxxsxxxxxsxssssssx
xxsssssxxxsssssxxssxxxssxssssssx
xsssxsssxsssxsssxsssxsssxssxxxxx
xssxxxssxssxxxssxsssssssxssxxxxx
xssxxxxxxssxxxssxsssssssxsssssxx
xssxxsssxsssssssxssxsxssxsssssxx
xssxxxssxsssssssxssxxxssxssxxxxx
xsssxsssxssxxxssxssxxxssxssxxxxx
xxsssssxxssxxxssxssxxxssxssssssx
xxxsssxxxssxxxssxssxxxssxssssssx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xx1xxx2xxx3xxxxxxxxxx3xxx2xxx1xx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxfxxfxxfxxfxxxxxxxxfxxfxxfxxfxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxsssxxxsxxxxxsxssssssxsssssxxx
xxsssssxxsxxxxxsxssssssxssssssxx
xsssxsssxssxxxssxssxxxxxssxxssxx
xssxxxssxssxxxssxssxxxxxssxxssxx
xssxxxssxsssxsssxsssssxxssssssxx
xssxxxssxxssxssxxsssssxxsssssxxx
xssxxxssxxsssssxxssxxxxxssxsssxx
xsssxsssxxxsssxxxssxxxxxssxxssxx
xxsssssxxxxsssxxxssssssxssxxssxx
xxxsssxxxxxxsxxxxssssssxssxxssxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
my_game.set_map_pattern('game-over', 'Lose', '1', '1', 32, 32)
my_game.add_map_segment('game-over', '1', lose_segment)
my_game.set_map_background('game-over', (60, 70, 88))
my_game.set_map_styles('game-over', 'ghost', { 'scale': 4 })
my_game.set_map_styles('game-over', 'pumpkin-ghost', { 'scale': 4 })
my_game.set_map_styles('game-over', 'alien-ghost', { 'scale': 4 })
my_game.set_map_styles('game-over', 'flame', { 'scale': (1.5, 2) })
my_game.set_map_styles('game-over', 'block', { 'color': (242, 249, 255) })
my_game.set_map_styles('game-over', 'spruce', { 'color': ((206, 228, 247), (161, 197, 230), (205, 219, 255)), 'scale': (1.2, 1.75) })

######### GAMEPLAY MAPS ########
# EASY lvl segments
# 	Map segments pattern
segments_pattern_easy = '''
21
34
'''
my_game.set_map_pattern('easy', 'Forest', segments_pattern_easy, '1', 14, 14)
my_game.set_map_background('easy', (130, 82, 2))
my_game.set_map_styles('easy', 'teleport', { 'colors': ((130, 82, 2), (255, 80, 80)) })
my_game.set_map_styles('easy', 'block', { 'color': (9, 158, 1) })

segments_easy_1 = '''
ssssssssssssss
mgmm______1ods
ssooosso_s_s1s
mgmms&hs_o_s_s
sso____s_s_o_s
mms_osso_s_s_s
omo_s_____ro_s
oms_s_s_osoo_s
sms_o_s_ok___s
oms_s_s_osoo_s
s_o___o____s_s
s_s_ooooss_o_s
o_o__mmomm_sms
s_sssomsmssoms
'''
# 	Segment 2
segments_easy_2 = '''
ssssssssssssss
s__________mmm
srs_o_ssossoos
sos_s_______mm
s__1sooss_ssos
sss_o__1o_stmm
sdo_o_s_s_odms
sgo_s_o_s_sdms
s_o_o_s_s_sdms
s_s_o_o_s_ssoo
smo___smo____s
smsgs_ogsssogs
smomskomsd_sms
smomosomoo_smo
'''
# 	Segment 3
segments_easy_3 = '''
smomsssmsogomo
s_oms__mm_mmms
s_o_s_soooss_o
s_o_____1o_mmg
s_sossos_s_oos
s________o_smm
ssoos_osso_o_s
s1__s_osso_s_s
s_s_o______smo
s_o_s_ssos__mm
s_s______o_sss
srossoos_o__ms
sdskmmm__o_omm
ssssssssssssss
'''
# 	Segment 4
segments_easy_4 = '''
sgoosomsmosoms
smmo__mmms___s
smss_oos_o_s_s
mmso_sdo_o_o_s
ssosms_o_s_s_s
_mmgms1____o_s
soos_o_ssosoos
s_os_o_______s
smos_sooos_sos
mmg__________s
s_s_so_ssoooss
s_o_s_1______s
mms___sooosoks
ssssssssssssss
'''
# Adding segments to the map
my_game.add_map_segment('easy', '1', segments_easy_1)
my_game.add_map_segment('easy', '2', segments_easy_2)
my_game.add_map_segment('easy', '3', segments_easy_3)
my_game.add_map_segment('easy', '4', segments_easy_4)

# MEDIUM lvl segments
# 	Map segments pattern
segments_pattern_easy = '''
456
123
'''
my_game.set_map_pattern('medium', 'Winter', segments_pattern_easy, '1', 16, 16)
my_game.set_map_background('medium', (60, 70, 88))
my_game.set_map_styles('medium', 'teleport', { 'colors': ((60, 70, 88), (255, 80, 80)) })
my_game.set_map_styles('medium', 'block', { 'color': (242, 249, 255) })
my_game.set_map_styles('medium', 'mark', { 'color': (149, 163, 185) })
my_game.set_map_styles('medium', 'spruce', { 'color': ((206, 228, 247), (161, 197, 230), (205, 219, 255)) })
my_game.set_map_styles('medium', 'ghost', { 'color': ((80, 163, 198), (121, 192, 215), (248, 248, 248), (221, 223, 223), (194, 194, 194)) })


# 	Segment 1
segments_medium_1 = '''
smssssssssmsssss
smsrmmmm2smskdds
smssgsssssgs_mmm
sdsmmm_____sssss
sgs_ss_sssssd_mm
s____s____ssssms
ssss_s_ss_s____s
s____s_s__s_ssss
s_s_ss_s_ss_smmm
s_s_s__s_s___mss
s1s___ss_s_ss_ss
sss_ssss_s_ss_ss
ss____s___1s__ms
s_&_s_s_ssss_smm
sh_ss________sms
ssssssssssssssss
'''
# 	Segment 2
segments_medium_2 = '''
smsssmssssmsssss
smsdsgs2ssgss_ds
gms_mmmmsmmmg_ss
sss_sss_s_s_s_ms
mms___s___s_ssmm
sms_s____ss__sss
s___ssss_sss___s
sss___s___ss_sss
mms_s_s_s__1___s
sm__s_s_s_s_ss_s
sssss___s_s_sd_s
s___sss1sks_ssss
sms_____sss_gmmm
mmsss_s_____ssss
smsr__sssss___ds
ssssssssssssssss
'''
# 	Segment 3
segments_medium_3 = '''
ssmsssssssgsssss
skms__2smmmsd__s
sdds_s_ssmssss_s
ssss_s_ss__s___s
mms__s__ss_s_sss
sms_sss____s___s
s_s_s___ss___s_s
s_____sss__sss_s
sssss_s___sss__s
s_____s_s_s___ss
s_sss_s_s___s_ss
sm__sm__sss_s_ss
mmsmgms_____s__s
sssmssss_s_sss_s
sd__sr___s____1s
ssssssssssssssss
'''
# 	Segment 4
segments_medium_4 = '''
ssssssssssssssss
sdss_s1__s____ds
s____s_sss_sssss
sss_ss_ss____gmm
s______sd_s_rsms
s_sss_sss_s_ss_s
s1__s____2s_ss_s
sss_ss_s_ss_g__s
ss___s_s_ss_ssss
ss_s___s__s____s
s__s_ssss_ssss_s
s_ss_______s___s
s_ds_ss_ssss_s_s
ssss_ss_s__s_s_s
smm___s__mmm_sks
smssssssssmsssss
'''
# 	Segment 5
segments_medium_5 = '''
ssssssssssssssss
s____ds_s1_____s
sms_sss_ssss_sds
mms___s__mmsmsss
sssss___ssgsmgmm
ss_ssms_s___msss
s___sms2s_ss_srs
s_s_sgsks_s____s
s_s_mmsss_s_ssss
s_sssms___s_ssmm
sds_____s_____ms
sss_s_sssss_ss_s
s___s____s___s_s
sss_ssss_s_s_s_s
smm_sm____ms__1s
smsssmssssmsssss
'''
# 	Segment 6
segments_medium_6 = '''
ssssssssssssssss
s1___s1__sss2sss
ssss_sss___sdtss
sm___s___s_gdd2s
mmss_ss_ssssgsss
s__mmgm_______ss
s_ssssms_ssss_ss
s___s__s____s__s
sss_s_sssss_ss_s
mms_s_________rs
sms___ss_sssssss
sms_s_s__s____2s
s___s_s_ss_s_s_s
sssss_s_ss_s_s_s
smmm_____smm_sks
ssmsssssssmsssss
'''
# Adding segments to the map
my_game.add_map_segment('medium', '1', segments_medium_1)
my_game.add_map_segment('medium', '2', segments_medium_2)
my_game.add_map_segment('medium', '3', segments_medium_3)
my_game.add_map_segment('medium', '4', segments_medium_4)
my_game.add_map_segment('medium', '5', segments_medium_5)
my_game.add_map_segment('medium', '6', segments_medium_6)


# HARD lvl segments
# 	Map segments pattern
segments_pattern_easy = '''
123
456
789
'''
my_game.set_map_pattern('hard', 'Desert', segments_pattern_easy, '7', 17, 17)
my_game.set_map_background('hard', (114, 94, 0))
my_game.set_map_styles('easy', 'teleport', { 'colors': ((114, 94, 0), (255, 80, 80)) })
my_game.set_map_styles('hard', 'block', { 'color': (255, 234, 135) })
ghosts_colors = ((230, 166, 68), (225, 131, 57), (182, 87, 29), (223, 145, 94), (243, 180, 139), (208, 183, 172))
my_game.set_map_styles('hard', 'pumpkin-ghost', { 'color': ghosts_colors })
my_game.set_map_styles('hard', 'ghost', { 'color': ghosts_colors })
my_game.set_map_styles('hard', 'teleport', { 'scale': 1.75 })

# 	Segment 1
segments_hard_1 = '''
bbbbbbbbbbbbbbbbb
bk_nlnn_b2_bbbbmb
bbbb1bb_bb____bmm
brbbbb___b_bb___b
blbb___bbb_bb_b_b
bnb__b___b__b_b_b
bn___b_b_bb___bbb
bbbb_b_b____b___b
b3___b_bb_bbbbb_b
bb_bbb__b___b1b_b
bb___bb_bbb_b_b_b
bn_b__b_____b___b
bnbbb___bb____bbb
bdb___bbb__b_mmmb
bbb_bbb___bbnbgbb
bmmm____bbbdnbmmm
bmbbbbbbbbbbbbmbb
'''
# 	Segment 2
segments_hard_2 = '''
bbbbbbbbbbbbbbbbb
bm____mdnlnb____b
mmbbb_bbbnbb_bbmb
bm__b_____b___bmm
bbb_b_bbbnl_1_bbb
bln___bbbbnbbn__b
b2bbb___rb_bdnb_b
bdb___b_bb_bbbb_b
bbb_bbb_______mmb
b_____bbb_bbnbbmb
b_bbb_____b_n2bgb
b___bb1bbbb_bnbmm
bmb_b_nln_b_bmbmb
bmb___bbb___bmgmb
bgbbbmbkbbb_bbbbb
mmdmgmblnn____bdm
bbbmbbbbbbbbbbbgb
'''
# 	Segment 3
segments_hard_3 = '''
bbbbbbbbbbbbbbbbb
b___bbbdln2b_bdlb
bbbm__bbbgbb_bbnb
mmgmb___mmm___b_b
bbbmbb_bb_b_b___b
bb___b_b__b_bbbnb
b__b_b_b_bb___bnb
bnbb__nb_b__b_n1b
bnb__b2n____bbbbb
bdb_bbbnbbb_bmm_b
bbb_b____b__bgb_b
mmm___bb_bb_mmb_b
bmbb__b___bb_bb_b
b_rbb_b_b____b3_b
bbbb__bnbbbbbbdbb
gmmm_bbl1bkmmbdbb
bbmbbbbbbbbbmbbbb
'''
# 	Segment 4
segments_hard_4 = '''
bmbbbbbbbbbbbbmbb
bmbm___bbrlnnbmbb
bmgmbb__bbbb_bdmm
bbb_bbb_b____bbgb
bbdnn_b_b_b___mmb
bbbbb_b___b_bbmbb
bnl3b_bb_bb_bbmbb
bnbn______b_mnmgm
bnb_bbbbb_bbbgbbb
bdb___bfb_bdnln2b
bbb_b_bbb_bbnbbbb
b___b______bllnnb
b_b_bb_b_b_bnbbnb
b_b__b_b_b_g2nbgb
b_bb___bnbbbbgbmm
bmmb_bbbl1bfbkddb
bbmbbbbbbbbbbbbbb
'''
# 	Segment 5
segments_hard_5 = '''
bbbmbbbbbbbbbbbmb
bdmmbnlndblnnbdmb
mmbbbnbbbbrb_bdmb
bbb_____bbbb_bbkb
b___b_bmmm__n1bbb
b_b_b_bbgbbbbnbmm
b_b_bbfl3lfb___mb
mmb_mbldddlbmb_bb
bmbbmg3dtd3gmb__b
b__b_bldddlbmbb_b
bbnb_bfl3lfb____b
bl1n__bbgbbbb_bbb
bdbbb__mmb____b_b
bbb___b_mm_bb___b
mmbbb_bbbb_bbbb_b
b__mb_mm___b_mm_b
bbbmbbbmbbbbbmbbb
'''
# 	Segment 6
segments_hard_6 = '''
bbmbbbbbbbbbmbbbb
b_mbnndb__2bmm_mb
b_bb_bbb_b_bbbbmb
bmm___b__b____bgb
bgbbb_b_bb_bb_mmb
mmbrb___b___b_bnb
bbblbb____b_b_bdb
b2nnnb_bbbb_bnbbb
bnbb______b_bnlbb
b__bb_bbb_b_bb1kb
bb__b_bfb____bbbb
bbb_b_bb___b____b
bdb_b____b_bbbbmb
bnb___bb_bbbmmgmb
bnb_bbb__nnb_bbbb
b______mbb2b__mmb
bbbbbbbmbbbbbbbmb
'''
# 	Segment 7
segments_hard_7 = '''
bbmbbbbbbbbbbbbbb
bbgb___bdnnbfb_hb
bmmm_b_bbbn_b_&_b
b_bb_b_________bb
b_bb_bbbbbb_b_bfb
b_b____bklb_b_mbb
b___bbnbbnb_bbmgm
bbbbbn2nb____bmbb
bdb__nb____b___bb
bnb_bbbb_b_bbb_bb
bnb_b____b______b
b2b_b_bbbb_bbbbnb
b______b___bbnn1b
b_bbbbmb_b___mbbb
bbb2bmm__bbbmmgmm
bdrlgmbb____mbbbb
bbbbbbbbbbbbbbbbb
'''
# 	Segment 8
segments_hard_8 = '''
bbbgbbbmbbbbbmbbb
bmmmgmbgb___bmgmm
b_bbbmmmb_b_bbbgb
b_bdbb____b___mmb
b_bnnn_bbbb_bbbbb
bmbbbb__b2nnb___b
mmgmbbb_blb___b_b
bbbmm___bkb_bbb_b
blnnb_b_bbb___b_b
bnb1b_b_____b___b
brbbb_bbb_bbbb_bb
bbb_____b_b____bb
b___bbb_____bmbbb
bmb___bbbb_bbmg2b
mmbbb_b1lb_b_mbnb
b_____bbnn_b_bbmm
bbbbbbbbbbbbbbbbb
'''
# 	Segment 9
segments_hard_9 = '''
bbbbbbbmbbbbbbbmb
mmb_bbbgblrbmm_mb
bmb___bmmnbbgbbbb
b___b__mbbbmmbdnb
bbbbb_bndb___bbnb
bdb___bnbb_b___3b
b1b_bbb____b_bbbb
blb_b___b_bb___bb
bnn_b_bbb_bbbb_bb
bbb_b_b__n2n___bb
b_b___b_bbnnb___b
b_b_bbb_b__bb_b_b
b____b__b_bb__bnb
b_bb___bb_bb_bbnb
bmmbbb_b_____b2lb
mm___b___bbb_bdkb
bbbbbbbbbbbbbbbbb
'''
# Adding segments to the map
my_game.add_map_segment('hard', '1', segments_hard_1)
my_game.add_map_segment('hard', '2', segments_hard_2)
my_game.add_map_segment('hard', '3', segments_hard_3)
my_game.add_map_segment('hard', '4', segments_hard_4)
my_game.add_map_segment('hard', '5', segments_hard_5)
my_game.add_map_segment('hard', '6', segments_hard_6)
my_game.add_map_segment('hard', '7', segments_hard_7)
my_game.add_map_segment('hard', '8', segments_hard_8)
my_game.add_map_segment('hard', '9', segments_hard_9)

# 	Running Game	
###########################
my_game.start()
game.start()
