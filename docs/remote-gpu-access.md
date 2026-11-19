# Reaching the Windows GPU machine over SSH

How the Mac was connected to the RTX 5090 PC (native Windows, no WSL) on 2026-11-19. The PC is
at another location, behind a router, with only a LAN address; AnyDesk was the only way in.
The result is `ssh <user>@<zerotier-ip>` from the Mac, with a key and no password.

Values specific to this setup (network ID, addresses, username) are left out of this file on
purpose. Placeholders are in `<angle brackets>`.

## What it takes

| Piece | Where | Why |
|---|---|---|
| OpenSSH server | PC | the way in |
| ZeroTier | PC and Mac | a private network between two machines on different networks |
| A firewall rule | PC | Windows blocks port 22 on the ZeroTier adapter by default |
| The Mac's public key | PC | login without a password |

Tailscale would do ZeroTier's job, but every Tailscale host reset the connection from this
network (2026-11-19), so it is not an option here.

## Procedure

Steps on the PC are run over AnyDesk, in an **administrator** PowerShell.

1. **PC: install and start the SSH server.**

   ```powershell
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service sshd -StartupType Automatic
   Get-Service sshd        # should say Running
   ```

2. **Any browser: create a ZeroTier network.** Sign up at `https://my.zerotier.com`, click
   **Create A Network**, and copy its 16-character network ID.

3. **PC: install ZeroTier and join.** Installer from `https://www.zerotier.com/download/`, then:

   ```powershell
   & "C:\Program Files (x86)\ZeroTier\One\zerotier-cli.bat" join <network-id>
   ```

   On the website, open the network, and under **Members** tick **Auth** for the new device.

4. **Mac: install ZeroTier and join.** In the Terminal app (it asks for the Mac password, so it
   cannot be run from Claude Code's `!` prompt):

   ```sh
   brew install --cask zerotier-one
   sudo zerotier-cli join <network-id>
   ```

   Tick **Auth** for the Mac on the website too.

5. **PC: open port 22 to the ZeroTier network only.** Use the subnet the website shows for the
   network.

   ```powershell
   New-NetFirewallRule -Name sshd-zerotier -DisplayName "OpenSSH over ZeroTier" -Direction Inbound -Protocol TCP -LocalPort 22 -RemoteAddress <zerotier-subnet>/24 -Action Allow -Profile Any
   ```

6. **PC: add the Mac's public key.** On the Mac, `cat ~/.ssh/id_ed25519.pub` prints it. For an
   account with administrator rights, Windows reads keys from this file, not from the user's
   own `.ssh` folder:

   ```powershell
   Add-Content C:\ProgramData\ssh\administrators_authorized_keys "<public key line>"
   icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
   whoami                  # prints computername\username; the username is what ssh needs
   ```

7. **Mac: connect.**

   ```sh
   ssh <user>@<zerotier-ip> nvidia-smi
   ```

   The PC's ZeroTier address is on the website's **Members** list.

## Checking each layer when it stops working

From the Mac, in this order:

| Check | Command | If it fails |
|---|---|---|
| The Mac is on the network | `ifconfig \| grep -A3 feth` shows an address in the subnet | ZeroTier is not running on the Mac, or the Mac lost its **Auth** tick |
| The PC is reachable | `ping <zerotier-ip>` | the PC is off, asleep, or ZeroTier is not running there |
| The SSH port answers | `nc -z -v -G 5 <zerotier-ip> 22` | the firewall rule (step 5) or the `sshd` service (step 1) |
| The key is accepted | `ssh -o BatchMode=yes <user>@<zerotier-ip> whoami` | the key file or its permissions (step 6) |

On 2026-11-19 the first three were how the firewall block was found: ping answered and port 22
timed out.

## Things to know

- **The link is slow.** Ping was about 360 ms, so ZeroTier is probably relaying. Commands are
  fine; multi-gigabyte files are not. Download models and pull code on the PC itself.
- **The PC must stay awake.** Set **Settings → System → Power → Sleep** to **Never**.
- **The SSH shell is Windows `cmd`**, not bash. PowerShell commands go through
  `powershell -Command "..."`. There is no `rsync`.
- **`zerotier-cli` on the Mac needs `sudo`**, so status checks there are run by a person.
- **Only access is set up.** The project has never been built on Windows. Building needs Rust
  (MSVC toolchain, with Visual Studio Build Tools), Git, Python 3 and the CUDA Toolkit, since the
  engine compiles its kernels at run time through NVRTC, which the driver alone does not include.

## Undoing it

- PC: `Remove-NetFirewallRule -Name sshd-zerotier`, `Stop-Service sshd`, and delete the key
  line from `C:\ProgramData\ssh\administrators_authorized_keys`.
- Either machine: leave the network from the ZeroTier tray or menu-bar icon, or delete the
  network on the website.
