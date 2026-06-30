-- config path:
-- ~/.config/wezterm/wezterm.lua in linux / macos
-- %USERPROFILE%\.config\wezterm\wezterm.lua in windows

--- __      __      _
--- \ \    / /__ __| |_ ___ _ _ _ __
---  \ \/\/ / -_)_ /  _/ -_) '_| '  \
---   \_/\_/\___/__|\__\___|_| |_|_|_|
---
--- wezterm.lua - personal configuration

local wezterm = require("wezterm")
-- local mux = wezterm.mux
local act = wezterm.action
local config = {}

-- CMD on macOS, CTRL on Linux / Windows
local PRIMARY_MOD = wezterm.target_triple:find("darwin") and "CMD" or "CTRL"

local function wezterm_binary()
  if wezterm.target_triple:find("windows") then
    return wezterm.executable_dir .. "\\wezterm.exe"
  end
  return wezterm.executable_dir .. "/wezterm"
end

-- Pane titles set from Lua (WezTerm has no pane:set_user_var API).
local custom_pane_titles = {}

local function build_merge_tab_choices(mux_window)
  local choices = {}
  for _, info in ipairs(mux_window:tabs_with_info()) do
    if not info.is_active then
      local tab = info.tab
      local active_pane = tab:active_pane()
      local title = active_pane:get_title() or tab:get_title() or "untitled"
      table.insert(choices, {
        id = tostring(active_pane:pane_id()),
        label = string.format("Tab %d: %s", info.index + 1, title),
      })
    end
  end
  return choices
end

local function merge_tab_as_vertical_split(window, pane)
  local mux_window = window:mux_window()
  if not mux_window then
    return
  end

  local choices = build_merge_tab_choices(mux_window)
  if #choices == 0 then
    window:toast_notification('merge tab', 'No other tabs to merge', nil, 3000)
    return
  end

  window:perform_action(
    act.InputSelector {
      title = "Select Tab to Convert into a Split Pane",
      fuzzy = true,
      choices = choices,
      action = wezterm.action_callback(function(_, inner_pane, id, _)
        if id then
          wezterm.run_child_process {
            wezterm_binary(),
            'cli', 'split-pane',
            '--move-pane-id', id,
            '--pane-id', tostring(inner_pane:pane_id()),
          }
        end
      end),
    },
    pane
  )
end

local function merge_tab_as_horizontal_split(window, pane)
  local mux_window = window:mux_window()
  if not mux_window then
    return
  end

  local choices = build_merge_tab_choices(mux_window)
  if #choices == 0 then
    window:toast_notification('merge tab', 'No other tabs to merge', nil, 3000)
    return
  end

  window:perform_action(
    act.InputSelector {
      title = "Select Tab to Convert into a Split Pane",
      fuzzy = true,
      choices = choices,
      action = wezterm.action_callback(function(_, inner_pane, id, _)
        if id then
          wezterm.run_child_process {
            wezterm_binary(),
            'cli', 'split-pane', '--horizontal',
            '--move-pane-id', id,
            '--pane-id', tostring(inner_pane:pane_id()),
          }
        end
      end),
    },
    pane
  )
end

local LEADER_HINT = 'Ctrl+A, then'
local PRIMARY_LABEL = PRIMARY_MOD == 'CMD' and 'Cmd' or 'Ctrl'

local action_close_pane = act.CloseCurrentPane { confirm = true }
local action_split_right = act.SplitVertical { domain = 'CurrentPaneDomain' }
local action_split_down = act.SplitHorizontal { domain = 'CurrentPaneDomain' }
local action_spawn_tab = act.SpawnTab 'CurrentPaneDomain'
local action_swap_pane = act.PaneSelect { mode = 'SwapWithActive' }
local action_rename_tab = act.PromptInputLine {
  description = 'Enter new name for tab',
  action = wezterm.action_callback(
    function(window, _, line)
      if line then
        window:active_tab():set_title(line)
      end
    end
  ),
}
local action_rename_pane = act.PromptInputLine {
  description = 'Enter new pane title:',
  action = wezterm.action_callback(function(window, pane, line)
    if line then
      custom_pane_titles[pane:pane_id()] = line
      window:set_right_status(build_right_status(window, pane))
    end
  end),
}

-- Use config builder object if possible
if wezterm.config_builder then
	config = wezterm.config_builder()
end

-- Use local unix domain multiplexer by default
config.default_mux_server_domain = 'local'
config.leader = { key = 'a', mods = 'CTRL', timeout_milliseconds = 10000}

-- Set colorscheme
config.color_scheme = "Banana Blueberry"

-- Font settings
-- config.font = wezterm.font("0xProto Nerd Font")
config.font_size = $FONT_SIZE$

config.window_frame = {
  font_size = $FONT_SIZE$,
}

config.inactive_pane_hsb = {
  saturation = 0.3,
  brightness = 0.3,
}

-- Window setting/ appearance
config.window_decorations = "TITLE|RESIZE"
config.window_background_opacity = 0.9
-- config.macos_window_background_blur = 10

config.window_padding = {
	left = "1cell",
	right = "1cell",
	top = "0.0cell",
	bottom = "0.5cell",
}

config.initial_rows = 40
config.initial_cols = 100

config.enable_scroll_bar = true
config.scrollback_lines = 5000
config.default_workspace = "main"

-- Dim inactive panes
--config.inactive_pane_hsb = {
--	saturation = 0.24,
--	brightness = 0.5,
--}

-- Tab bar
-- config.enable_tab_bar = false
config.use_fancy_tab_bar = true
config.tab_bar_at_bottom = false
config.status_update_interval = 200

local function basename(s)
  return string.gsub(s, "(.*[/\\])(.*)", "%2")
end

local function build_right_status(window, pane)
  local cwd = pane:get_current_working_dir()
  if cwd then
    if type(cwd) == "userdata" then
      cwd = basename(cwd.file_path)
    else
      cwd = basename(cwd)
    end
  else
    cwd = ""
  end

  local cmd = pane:get_foreground_process_name()
  cmd = cmd and basename(cmd) or ""

  local status = {}

  if window:leader_is_active() then
    table.insert(status, { Attribute = { Intensity = "Bold" } })
    table.insert(status, { Background = { Color = "#e0af68" } })
    table.insert(status, { Foreground = { Color = "#1a1b26" } })
    table.insert(status, { Text = " ⚡ LEADER WAITING " })
    table.insert(status, "ResetAttributes")
    table.insert(status, { Text = " | " })
  end

  table.insert(status, { Text = wezterm.nerdfonts.md_folder .. "  " .. cwd })
  table.insert(status, { Text = " | " })
  table.insert(status, { Foreground = { Color = "#e0af68" } })
  table.insert(status, { Text = wezterm.nerdfonts.fa_code .. "  " .. cmd })
  table.insert(status, "ResetAttributes")
  table.insert(status, { Text = "  " })

  return wezterm.format(status)
end

wezterm.on("update-status", function(window, pane)
	window:set_right_status(build_right_status(window, pane))
end)

wezterm.on('format-tab-title', function(tab, tabs, panes, config, hover, max_width)
  local pane_id = tab.active_pane.pane_id
  local pane_title = tab.active_pane.title

  local user_title = custom_pane_titles[pane_id]
    or tab.active_pane.user_vars.custom_pane_title
  if user_title ~= nil and #user_title > 0 then
    pane_title = user_title
  end

  local tab_title = tab.tab_title
  if tab_title ~= nil and #tab_title > 0 then
    pane_title = tab_title
  end

  return string.format(" [%d] %s ", pane_id, pane_title)
end)

local keys = {}

local function disable_default(key, mods)
  table.insert(keys, { key = key, mods = mods, action = act.DisableDefaultAssignment })
end

-- Clear stale default bindings so Ctrl+Shift+P shows our custom shortcuts.
-- disable_default('t', 'SUPER')
-- disable_default('t', 'CTRL|SHIFT')
-- disable_default('w', 'SUPER')
-- disable_default('w', 'CTRL|SHIFT')
-- disable_default('"', 'CTRL|SHIFT|ALT')
-- disable_default('%', 'CTRL|SHIFT|ALT')
-- disable_default("'", 'CTRL|SHIFT|ALT')
-- disable_default('5', 'CTRL|SHIFT|ALT')
-- disable_default('"', 'ALT|CTRL')
-- disable_default('%', 'ALT|CTRL')
-- disable_default("'", 'ALT|CTRL')
-- disable_default('5', 'ALT|CTRL')

local function bind(key, mods, action)
  table.insert(keys, { key = key, mods = mods, action = action })
end

-- Windows / Linux
-- Move cursor word-by-word (Left / Right)
bind('LeftArrow', 'CTRL', act.SendString '\x1bb')
bind('RightArrow', 'CTRL', act.SendString '\x1bf')
-- Move to the START / END of the line
bind('a', 'ALT', act.SendString '\x1bb')
bind('e', 'ALT', act.SendString '\x1bf')
-- Delete a WORD backwards
bind('w', 'ALT', act.SendString '\x1b\x7f')
-- Clear screen and discard the scrollback buffer
bind('k', 'CTRL', act.ClearScrollback 'ScrollbackAndViewport')

-- MacOS
-- Move cursor word-by-word (Left / Right)
bind('LeftArrow', 'OPT', act.SendKey { key = 'b', mods = 'ALT' })
bind('RightArrow', 'OPT', act.SendKey { key = 'f', mods = 'ALT' })
-- Clear screen and scrollback buffer using Command + K
bind('k', 'CMD', act.ClearScrollback 'ScrollbackAndViewport')

-- Pane management (primary modifier: CMD on macOS, CTRL on Linux / Windows)
bind('w', PRIMARY_MOD .. '|SHIFT', action_close_pane)
bind('d', PRIMARY_MOD, action_split_right)
bind('d', PRIMARY_MOD .. '|SHIFT', action_split_down)

-- Spawn a new tab
bind('t', 'LEADER', action_spawn_tab)
-- Swap active pane with a selected pane
bind('s', 'LEADER', action_swap_pane)
-- Press LEADER then 'v' or 'h' to pull another tab's pane into the current tab as a split
bind('v', 'LEADER', wezterm.action_callback(merge_tab_as_vertical_split))
bind('h', 'LEADER', wezterm.action_callback(merge_tab_as_horizontal_split))
-- Close the current pane
bind('x', 'LEADER', action_close_pane)
-- Change the title of the current tab
bind('e', 'LEADER', action_rename_tab)
bind('r', 'LEADER', action_rename_pane)

wezterm.on('augment-command-palette', function(_window, _pane)
  return {
    {
      brief = 'Pane: Close current',
      doc = PRIMARY_LABEL .. '+Shift+W, or ' .. LEADER_HINT .. ' x',
      icon = 'md_close',
      action = action_close_pane,
    },
    {
      brief = 'Pane: Split to the right',
      doc = PRIMARY_LABEL .. '+D, or ' .. LEADER_HINT .. ' Shift+%',
      icon = 'md_view_column',
      action = action_split_right,
    },
    {
      brief = 'Pane: Split downward',
      doc = PRIMARY_LABEL .. '+Shift+D, or ' .. LEADER_HINT .. ' Shift+"',
      icon = 'md_view_agenda',
      action = action_split_down,
    },
    {
      brief = 'Pane: Swap with selected pane',
      doc = LEADER_HINT .. ' s',
      icon = 'md_swap_horizontal',
      action = action_swap_pane,
    },
    {
      brief = 'Pane: Focus left',
      doc = LEADER_HINT .. ' j',
      icon = 'md_arrow_left',
      action = act.ActivatePaneDirection 'Left',
    },
    {
      brief = 'Pane: Focus right',
      doc = LEADER_HINT .. ' l',
      icon = 'md_arrow_right',
      action = act.ActivatePaneDirection 'Right',
    },
    {
      brief = 'Pane: Focus up',
      doc = LEADER_HINT .. ' i',
      icon = 'md_arrow_up',
      action = act.ActivatePaneDirection 'Up',
    },
    {
      brief = 'Pane: Focus down',
      doc = LEADER_HINT .. ' k',
      icon = 'md_arrow_down',
      action = act.ActivatePaneDirection 'Down',
    },
    {
      brief = 'Tab: Merge into vertical split',
      doc = LEADER_HINT .. ' v',
      icon = 'md_view_column',
      action = wezterm.action_callback(merge_tab_as_vertical_split),
    },
    {
      brief = 'Tab: Merge into horizontal split',
      doc = LEADER_HINT .. ' h',
      icon = 'md_view_agenda',
      action = wezterm.action_callback(merge_tab_as_horizontal_split),
    },
    {
      brief = 'Tab: Spawn new',
      doc = LEADER_HINT .. ' t',
      icon = 'md_tab_plus',
      action = action_spawn_tab,
    },
    {
      brief = 'Tab: Rename',
      doc = LEADER_HINT .. ' e',
      icon = 'md_rename_box',
      action = action_rename_tab,
    },
    {
      brief = 'Pane: Rename title',
      doc = LEADER_HINT .. ' r',
      icon = 'md_form_textbox',
      action = action_rename_pane,
    },
    {
      brief = 'Shell: Move cursor word left',
      doc = 'Ctrl+Left (Linux/Windows), Option+Left (macOS)',
      icon = 'md_keyboard',
      action = act.SendString '\x1bb',
    },
    {
      brief = 'Shell: Move cursor word right',
      doc = 'Ctrl+Right (Linux/Windows), Option+Right (macOS)',
      icon = 'md_keyboard',
      action = act.SendString '\x1bf',
    },
    {
      brief = 'Shell: Move cursor to line start',
      doc = 'Alt+A',
      icon = 'md_keyboard',
      action = act.SendString '\x1bb',
    },
    {
      brief = 'Shell: Move cursor to line end',
      doc = 'Alt+E',
      icon = 'md_keyboard',
      action = act.SendString '\x1bf',
    },
    {
      brief = 'Shell: Delete word backward',
      doc = 'Alt+W',
      icon = 'md_backspace',
      action = act.SendString '\x1b\x7f',
    },
    {
      brief = 'Shell: Clear scrollback and screen',
      doc = PRIMARY_LABEL .. '+K (macOS), Ctrl+K (Linux/Windows)',
      icon = 'md_broom',
      action = act.ClearScrollback 'ScrollbackAndViewport',
    },
  }
end)

config.keys = keys

-- Return the configuration to wezterm
return config
