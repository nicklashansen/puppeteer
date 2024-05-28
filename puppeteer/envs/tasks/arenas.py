import os

import numpy as np
from dm_control import composer
from dm_control.composer import variation
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
from dm_control.locomotion.arenas.corridors import Corridor


_SIDE_WALLS_GEOM_GROUP = 3
_CORRIDOR_X_PADDING = 2.0
_GROUNDPLANE_QUAD_SIZE = 0.25
_WALL_THICKNESS = 0.16
_SIDE_WALL_HEIGHT = 4.0
_DEFAULT_ALPHA = 0.5


class EmptyCorridor(Corridor):
  """An empty corridor with planes around the perimeter."""

  def _build(self,
             corridor_width=5,
             corridor_length=40,
             visible_side_planes=True,
             name='empty_corridor'):
    """Builds the corridor.

    Args:
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      name: The name of this arena.
    """
    super()._build(name=name)

    self._corridor_width = corridor_width
    self._corridor_length = corridor_length

    self._walls_body = self._mjcf_root.worldbody.add('body', name='walls')

    self._mjcf_root.visual.map.znear = 0.0005
    self._mjcf_root.asset.add(
        'texture', type='skybox', builtin='gradient',
        rgb1=[.9, .9, .9], rgb2=[.9, .9, .9], width=100, height=600)
    self._mjcf_root.visual.headlight.set_attributes(
        ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8],
        specular=[0.1, 0.1, 0.1])

    alpha = _DEFAULT_ALPHA if visible_side_planes else 0.0
    self._ground_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', rgba=[.8, .8, .8, 1], size=[1, 1, 1])
    self._left_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[1, 0, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
    self._right_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[-1, 0, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
    self._near_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[0, 1, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
    self._far_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[0, -1, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)

    self._current_corridor_length = None
    self._current_corridor_width = None

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor is resized accordingly.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    self._walls_body.geom.clear()
    corridor_width = variation.evaluate(self._corridor_width,
                                        random_state=random_state)
    corridor_length = variation.evaluate(self._corridor_length,
                                         random_state=random_state)
    self._current_corridor_length = corridor_length
    self._current_corridor_width = corridor_width

    self._ground_plane.pos = [corridor_length / 2, 0, 0]
    self._ground_plane.size = [
        corridor_length / 2 + _CORRIDOR_X_PADDING, corridor_width / 2, 1]

    self._left_plane.pos = [
        corridor_length / 2, corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
    self._left_plane.size = [
        corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

    self._right_plane.pos = [
        corridor_length / 2, -corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
    self._right_plane.size = [
        corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

    self._near_plane.pos = [
        -_CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
    self._near_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

    self._far_plane.pos = [
        corridor_length + _CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
    self._far_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

  @property
  def corridor_length(self):
    return self._current_corridor_length

  @property
  def corridor_width(self):
    return self._current_corridor_width

  @property
  def ground_geoms(self):
    return (self._ground_plane,)


class GapsCorridor(EmptyCorridor):
  """A corridor that consists of multiple platforms separated by gaps."""

  # pylint: disable=arguments-renamed
  def _build(self,
             platform_length=1.,
             gap_length=2.5,
             corridor_width=5,
             corridor_length=40,
             ground_rgba=(.8, .8, .8, 1),
             visible_side_planes=False,
             aesthetic='default',
             name='gaps_corridor'):
    """Builds the corridor.

    Args:
      platform_length: A number or a `composer.variation.Variation` object that
        specifies the size of the platforms along the corridor.
      gap_length: A number or a `composer.variation.Variation` object that
        specifies the size of the gaps along the corridor.
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      ground_rgba: A sequence of 4 numbers or a `composer.variation.Variation`
        object specifying the color of the ground.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      aesthetic: option to adjust the material properties and skybox
      name: The name of this arena.
    """
    super()._build(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes,
        name=name)

    self._platform_length = platform_length
    self._gap_length = gap_length
    self._ground_rgba = ground_rgba
    self._aesthetic = aesthetic

    if self._aesthetic != 'default':
      ground_info = locomotion_arenas_assets.get_ground_texture_info(aesthetic)
      sky_info = locomotion_arenas_assets.get_sky_texture_info(aesthetic)
      texturedir = locomotion_arenas_assets.get_texturedir(aesthetic)
      self._mjcf_root.compiler.texturedir = texturedir

      self._ground_texture = self._mjcf_root.asset.add(
          'texture', name='aesthetic_texture', file=ground_info.file,
          type=ground_info.type)
      self._ground_material = self._mjcf_root.asset.add(
          'material', name='aesthetic_material', texture=self._ground_texture,
          texuniform='true')
      # remove existing skybox
      for texture in self._mjcf_root.asset.find_all('texture'):
        if texture.type == 'skybox':
          texture.remove()
      self._skybox = self._mjcf_root.asset.add(
          'texture', name='aesthetic_skybox', file=sky_info.file,
          type='skybox', gridsize=sky_info.gridsize,
          gridlayout=sky_info.gridlayout)

    self._ground_body = self._mjcf_root.worldbody.add('body', name='ground')

  # pylint: enable=arguments-renamed

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor resized accordingly, and
    new sets of platforms are created according to values drawn from the
    `platform_length`, `gap_length`, and `ground_rgba` distributions specified
    in `_build`.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    # Resize the entire corridor first.
    super().regenerate(random_state)

    # Move the ground plane down and make it invisible.
    self._ground_plane.pos = [self._current_corridor_length / 2, 0, -10]
    self._ground_plane.rgba = [0, 0, 0, 0]

    # Clear the existing platform pieces.
    self._ground_body.geom.clear()

    # Make the first platform larger.
    platform_length = 3. * _CORRIDOR_X_PADDING
    platform_pos = [0, 0, -_WALL_THICKNESS]
    platform_size = [
        platform_length / 2,
        self._current_corridor_width / 2,
        _WALL_THICKNESS,
    ]
    if self._aesthetic != 'default':
      self._ground_body.add(
          'geom',
          type='box',
          name='start_floor',
          pos=platform_pos,
          size=platform_size,
          material=self._ground_material)
    else:
      self._ground_body.add(
          'geom',
          type='box',
          rgba=variation.evaluate(self._ground_rgba, random_state),
          name='start_floor',
          pos=platform_pos,
          size=platform_size)

    current_x = platform_length / 2
    platform_id = 0
    while current_x < self._current_corridor_length:
      platform_length = variation.evaluate(
          self._platform_length, random_state=random_state)
      platform_pos = [
          current_x + platform_length / 2.,
          0,
          -_WALL_THICKNESS,
      ]
      platform_size = [
          platform_length / 2,
          self._current_corridor_width / 2,
          _WALL_THICKNESS,
      ]
      if self._aesthetic != 'default':
        self._ground_body.add(
            'geom',
            type='box',
            name='floor_{}'.format(platform_id),
            pos=platform_pos,
            size=platform_size,
            material=self._ground_material)
      else:
        self._ground_body.add(
            'geom',
            type='box',
            rgba=variation.evaluate(self._ground_rgba, random_state),
            name='floor_{}'.format(platform_id),
            pos=platform_pos,
            size=platform_size)

      platform_id += 1

      # Move x to start of the next platform.
      current_x += platform_length + variation.evaluate(
          self._gap_length, random_state=random_state)

  @property
  def ground_geoms(self):
    return (self._ground_plane,) + tuple(self._ground_body.find_all('geom'))


class WallsCorridor(EmptyCorridor):
  """A corridor obstructed by multiple walls aligned against the two sides."""

  # pylint: disable=arguments-renamed
  def _build(self,
             wall_gap=2.5,
             wall_width=2.5,
             wall_height=2.0,
             swap_wall_side=True,
             wall_rgba=(1, 1, 1, 1),
             corridor_width=5,
             corridor_length=40,
             visible_side_planes=False,
             include_initial_padding=True,
             name='walls_corridor'):
    """Builds the corridor.

    Args:
      wall_gap: A number or a `composer.variation.Variation` object that
        specifies the gap between each consecutive pair obstructing walls.
      wall_width: A number or a `composer.variation.Variation` object that
        specifies the width that the obstructing walls extend into the corridor.
      wall_height: A number or a `composer.variation.Variation` object that
        specifies the height of the obstructing walls.
      swap_wall_side: A boolean or a `composer.variation.Variation` object that
        specifies whether the next obstructing wall should be aligned against
        the opposite side of the corridor compared to the previous one.
      wall_rgba: A sequence of 4 numbers or a `composer.variation.Variation`
        object specifying the color of the walls.
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      include_initial_padding: Whether to include initial offset before first
        obstacle.
      name: The name of this arena.
    """
    super()._build(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes,
        name=name)

    self._wall_height = wall_height
    self._wall_rgba = wall_rgba
    self._wall_gap = wall_gap
    self._wall_width = wall_width
    self._swap_wall_side = swap_wall_side
    self._include_initial_padding = include_initial_padding

  # pylint: enable=arguments-renamed

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor resized accordingly, and
    new sets of obstructing walls are created according to values drawn from the
    `wall_gap`, `wall_width`, `wall_height`, and `wall_rgba` distributions
    specified in `_build`.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    super().regenerate(random_state)

    wall_x = variation.evaluate(
        self._wall_gap, random_state=random_state) - _CORRIDOR_X_PADDING
    if self._include_initial_padding:
      wall_x += 2*_CORRIDOR_X_PADDING
    wall_side = 0
    wall_id = 0
    while wall_x < self._current_corridor_length:
      wall_width = variation.evaluate(
          self._wall_width, random_state=random_state)
      wall_height = variation.evaluate(
          self._wall_height, random_state=random_state)
      wall_rgba = variation.evaluate(self._wall_rgba, random_state=random_state)
      if variation.evaluate(self._swap_wall_side, random_state=random_state):
        wall_side = 1 - wall_side

      wall_pos = [
          wall_x,
          (2 * wall_side - 1) * (self._current_corridor_width - wall_width) / 2,
          wall_height / 2
      ]
      wall_size = [_WALL_THICKNESS / 2, wall_width / 2, wall_height / 2]
      self._walls_body.add(
          'geom',
          type='box',
          name='wall_{}'.format(wall_id),
          pos=wall_pos,
          size=wall_size,
          rgba=wall_rgba)

      wall_id += 1
      wall_x += variation.evaluate(self._wall_gap, random_state=random_state)

  @property
  def ground_geoms(self):
    return (self._ground_plane,)


class StairsCorridor(EmptyCorridor):
  """A corridor with stairs of increasing height."""

  # pylint: disable=arguments-renamed
  def _build(self,
             stair_length=2.,
             stair_height=0.2,
             corridor_width=5,
             corridor_length=40,
             visible_side_planes=False,
             include_initial_padding=True,
             name='stairs_corridor'):
    """Builds the corridor.

    Args:
      stair_length: A number or a `composer.variation.Variation` object that
        specifies the length of each stair.
      stair_height: A number or a `composer.variation.Variation` object that
        specifies the height of each stair.
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      include_initial_padding: Whether to include initial offset before first
        obstacle.
      name: The name of this arena.
    """
    super()._build(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes,
        name=name)

    self._stair_length = stair_length
    self._stair_height = stair_height
    self._include_initial_padding = include_initial_padding

  # pylint: enable=arguments-renamed

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor resized accordingly, and
    new sets of stairs are created according to values drawn from the
    `stair_length`, and `stair_height` distributions specified in `_build`.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    super().regenerate(random_state)

    stair_length = variation.evaluate(self._stair_length,
                                      random_state=random_state)
    stair_height = variation.evaluate(self._stair_height,
                                      random_state=random_state)

    stair_x = stair_length - _CORRIDOR_X_PADDING
    if self._include_initial_padding:
      stair_x += 2*_CORRIDOR_X_PADDING
    stair_z = stair_height / 2

    stair_id = 0
    while stair_x < self._current_corridor_length:
      stair_pos = [stair_x, 0, stair_z]
      stair_size = [stair_length / 2, self._current_corridor_width / 2,
                    stair_height / 2]
      self._walls_body.add(
          'geom',
          type='box',
          name='stair_{}'.format(stair_id),
          pos=stair_pos,
          size=stair_size,
          rgba=[.75, .75, .75, 1])

      stair_x += stair_length
      stair_z += stair_height
      stair_id += 1

  @property
  def ground_geoms(self):
    return (self._ground_plane,)


class HurdlesCorridor(EmptyCorridor):
  """A corridor with randomly placed hurdles."""

  # pylint: disable=arguments-renamed
  def _build(self,
             hurdle_length=0.1,
             hurdle_height=0.2,
             hurdle_spacing=2.0,
             corridor_width=5,
             corridor_length=40,
             visible_side_planes=False,
             include_initial_padding=True,
             name='stairs_corridor'):
    """Builds the corridor.

    Args:
      hurdle_length: A number or a `composer.variation.Variation` object that
        specifies the length of each hurdle.
      hurdle_height: A number or a `composer.variation.Variation` object that
        specifies the height of each hurdle.
      hurdle_spacing: A number or a `composer.variation.Variation` object that
        specifies the spacing between each hurdle.
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      include_initial_padding: Whether to include initial offset before first
        obstacle.
      name: The name of this arena.
    """
    super()._build(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes,
        name=name)

    self._hurdle_length = hurdle_length
    self._hurdle_height = hurdle_height
    self._hurdle_spacing = hurdle_spacing
    self._include_initial_padding = include_initial_padding

  # pylint: enable=arguments-renamed

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor resized accordingly, and
    new sets of stairs are created according to values drawn from the
    `stair_length`, and `stair_height` distributions specified in `_build`.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    super().regenerate(random_state)

    hurdle_length = variation.evaluate(self._hurdle_length,
                                      random_state=random_state)
    hurdle_height = variation.evaluate(self._hurdle_height,
                                      random_state=random_state)
    hurdle_spacing = variation.evaluate(self._hurdle_spacing,
                                      random_state=random_state)

    hurdle_x = hurdle_length - _CORRIDOR_X_PADDING
    if self._include_initial_padding:
      hurdle_x += 2*_CORRIDOR_X_PADDING
    hurdle_z = hurdle_height / 2

    hurdle_id = 0
    while hurdle_x < self._current_corridor_length:
      hurdle_pos = [hurdle_x, 0, hurdle_z]
      hurdle_size = [hurdle_length / 2, self._current_corridor_width / 2,
                    hurdle_height / 2]
      self._walls_body.add(
          'geom',
          type='box',
          name='hurdle_{}'.format(hurdle_id),
          pos=hurdle_pos,
          size=hurdle_size,
          rgba=[.7, .3, .5, 1])

      hurdle_x += hurdle_length + hurdle_spacing
      hurdle_id += 1

  @property
  def ground_geoms(self):
    return (self._ground_plane,)


class Floor(composer.Arena):
  """A simple floor arena with a customized look."""

  def _build(self, size=(8, 8), reflectance=0., name='floor',
             top_camera_y_padding_factor=1.1, top_camera_distance=100):
    super()._build(name=name)
    self._size = size
    self._top_camera_y_padding_factor = top_camera_y_padding_factor
    self._top_camera_distance = top_camera_distance

    self._mjcf_root.visual.headlight.set_attributes(
        ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

    # Add checkered floor
    self._ground_texture = self._mjcf_root.asset.add(
        'texture',
        rgb1=[.85, .85, .85],
        rgb2=[.80, .80, .80],
        type='2d',
        builtin='checker',
        name='groundplane',
        width=200,
        height=200)
    self._ground_material = self._mjcf_root.asset.add(
        'material',
        name='groundplane',
        texrepeat=[0.5, 0.5],
        texuniform=True,
        reflectance=reflectance,
        texture=self._ground_texture)
    self._ground_geom = self._mjcf_root.worldbody.add(
        'geom',
        type='plane',
        name='groundplane',
        material=self._ground_material,
        size=list(size) + [_GROUNDPLANE_QUAD_SIZE])
    
    # Add skybox
    self._skybox_texture = self._mjcf_root.asset.add(
      'texture',
      type='skybox',
      builtin='gradient',
      rgb1=[.9, .9, .9],
      rgb2=[.9, .9, .9],
      width=100,
      height=100)

    # Choose the FOV so that the floor always fits nicely within the frame
    # irrespective of actual floor size.
    fovy_radians = 2 * np.arctan2(top_camera_y_padding_factor * size[1],
                                  top_camera_distance)
    self._top_camera = self._mjcf_root.worldbody.add(
        'camera',
        name='top_camera',
        pos=[0, 0, top_camera_distance],
        quat=[1, 0, 0, 0],
        fovy=np.rad2deg(fovy_radians))

  @property
  def ground_geoms(self):
    return (self._ground_geom,)

  def regenerate(self, random_state):
    pass

  @property
  def size(self):
    return self._size


class PolesCorridor(EmptyCorridor):
    """A corridor with randomly placed poles."""
    def _build(self, pole_radius=0.05, pole_height=2.0, pole_position=0.0,
               corridor_width=5, corridor_length=40, visible_side_planes=False,
               include_initial_padding=True, name='poles_corridor'):
        super()._build(corridor_width=corridor_width,
                       corridor_length=corridor_length,
                       visible_side_planes=visible_side_planes,
                       name=name)
        self._pole_radius = pole_radius
        self._pole_height = pole_height
        self._pole_position = pole_position
        self._include_initial_padding = include_initial_padding

    def regenerate(self, random_state):
        """Regenerates this corridor with randomly placed poles."""
        super().regenerate(random_state)

        # Evaluate the variations
        pole_radius = variation.evaluate(self._pole_radius, random_state=random_state)
        pole_height = variation.evaluate(self._pole_height, random_state=random_state)
        pole_id = 1
        pole_x = 0 if self._include_initial_padding else -_CORRIDOR_X_PADDING

        # Calculate initial x position based on padding settings
        if self._include_initial_padding:
            pole_x += 2 * _CORRIDOR_X_PADDING

        # Gap between rows of poles, hardcoded as 2.0
        gap_between_rows = 1.0
        rows = int(self._current_corridor_length / gap_between_rows)
        num_poles_per_row = 2
        
        # Place poles in rows across the corridor length
        for i in range(1, rows + 1):
            x_position = i * gap_between_rows

            positions = [(x_position, variation.evaluate(self._pole_position,random_state), 0) for _ in range(num_poles_per_row)]
            for pos in positions:
                self._walls_body.add(
                    'geom', type='cylinder', name=f'poles_{pole_id}',
                    pos=pos, size=[pole_radius, pole_height],
                    friction=[0.6, 0.005, 0.0001],
                    rgba=[1., 0.6, 0.6, 1])
                pole_id += 1


class SlidesCorridor(EmptyCorridor):
    """A corridor with randomly placed slides."""
    def _build(self, slide_length=5.0, slide_height=0.05, slide_spacing=2.0,
               corridor_width=5, corridor_length=40, visible_side_planes=False,
               include_initial_padding=True, name='slides_corridor'):
        super()._build(corridor_width=corridor_width,
                       corridor_length=corridor_length,
                       visible_side_planes=visible_side_planes,
                       name=name)
        self._slides = []
        self._slide_length = slide_length
        self._slide_height = slide_height
        self._slide_spacing = slide_spacing
        self._include_initial_padding = include_initial_padding
        # Load slide mesh
        file_path = os.path.join(os.path.dirname(__file__), 'slide.stl')
        for i in range(0,7):
            # slides of different heights
            height = i * 0.01
            self.mjcf_model.asset.add('mesh', file=file_path, name=f'slide_mesh_{i}', scale=[0.2, 0.26, height])

    def regenerate(self, random_state):
        """Regenerates the corridor with new slides."""
        super().regenerate(random_state=random_state)
        slide_height = variation.evaluate(self._slide_height, random_state=random_state)
        # quantize the height to 0.01
        slide_height = round(slide_height, 2)+0.01
        slide_length = variation.evaluate(self._slide_length, random_state=random_state)
        slide_x = 0 if not self._include_initial_padding else 0 * slide_length
        i_d = 0

        while slide_x < self._corridor_length:
            i_d += 1
            slide_pos = [slide_x, -5.3, 0]
            mesh_name = f'slide_mesh_{int(slide_height*100)}'
            self._walls_body.add('geom',
                                          type='mesh',
                                          mesh=mesh_name,
                                          name='slide_slide_{}'.format(i_d),
                                          pos=slide_pos,
                                          # size=[1, 1, 10],
                                          rgba=[1.0, 0.6, 0.6, 1])

            slide_x += slide_length
            slide_length = variation.evaluate(self._slide_length, random_state=random_state)


    @property
    def ground_geoms(self):
        """Returns the ground geometries of the corridor."""
        return (self._ground_plane,)
