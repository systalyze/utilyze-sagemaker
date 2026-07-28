package procfs

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type Stat struct {
	PID       int
	Comm      string
	State     byte
	PPID      int
	PGID      int
	SessionID int
	StartTime uint64 // field 22 ("starttime"): clock ticks since boot that survives PID reuse
}

// Comm returns /proc/<pid>/comm.
func Comm(pid int) (string, error) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/comm", pid))
	if err != nil {
		return "", err
	}
	return strings.TrimRight(string(raw), "\n"), nil
}

// Cmdline returns the argv of a running process, split on NUL bytes.
// Zombie processes return an empty slice with no error.
func Cmdline(pid int) ([]string, error) {
	raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		return nil, nil
	}
	if raw[len(raw)-1] == 0 {
		raw = raw[:len(raw)-1]
	}
	return strings.Split(string(raw), "\x00"), nil
}

// StatForPID parses /proc/<pid>/stat.
func StatForPID(pid int) (Stat, error) {
	data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if err != nil {
		return Stat{}, err
	}
	return parseStat(string(data))
}

func parseStat(raw string) (Stat, error) {
	// /proc/.../stat is: pid (comm) state ppid pgrp session ... starttime ...
	// comm is wrapped in parens and may contain spaces and ) chars; the LAST
	// ')' ends the comm field. Everything after that is whitespace-delimited.
	open := strings.IndexByte(raw, '(')
	close := strings.LastIndexByte(raw, ')')
	if open < 0 || close < 0 || close < open {
		return Stat{}, errors.New("procfs: malformed stat payload")
	}

	pidStr := strings.TrimSpace(raw[:open])
	parsedPID, err := strconv.Atoi(pidStr)
	if err != nil {
		return Stat{}, fmt.Errorf("procfs: bad pid %q: %w", pidStr, err)
	}

	comm := raw[open+1 : close]
	rest := strings.Fields(strings.TrimSpace(raw[close+1:]))
	if len(rest) < 20 {
		return Stat{}, errors.New("procfs: short stat payload")
	}

	ppid, err := strconv.Atoi(rest[1])
	if err != nil {
		return Stat{}, fmt.Errorf("procfs: bad ppid %q: %w", rest[1], err)
	}
	pgid, err := strconv.Atoi(rest[2])
	if err != nil {
		return Stat{}, fmt.Errorf("procfs: bad pgid %q: %w", rest[2], err)
	}
	sessionID, err := strconv.Atoi(rest[3])
	if err != nil {
		return Stat{}, fmt.Errorf("procfs: bad session id %q: %w", rest[3], err)
	}
	startTime, err := strconv.ParseUint(rest[19], 10, 64)
	if err != nil {
		return Stat{}, fmt.Errorf("procfs: bad starttime %q: %w", rest[19], err)
	}

	return Stat{
		PID:       parsedPID,
		Comm:      comm,
		State:     rest[0][0],
		PPID:      ppid,
		PGID:      pgid,
		SessionID: sessionID,
		StartTime: startTime,
	}, nil
}

// SessionPeers enumerates /proc for PIDs sharing the given session ID.
func SessionPeers(sessionID int) ([]int, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}

	var peers []int
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		stat, err := StatForPID(pid)
		if err != nil {
			continue
		}
		if stat.SessionID == sessionID {
			peers = append(peers, pid)
		}
	}
	return peers, nil
}

// ListeningPorts returns the local TCP ports on which the given PID is
// currently listening.
func ListeningPorts(pid int) ([]int, error) {
	inodes, err := readSocketInodes(pid)
	if err != nil {
		return nil, err
	}
	if len(inodes) == 0 {
		return nil, nil
	}

	var ports []int
	seen := make(map[int]bool)
	for _, path := range []string{
		fmt.Sprintf("/proc/%d/net/tcp", pid),
		fmt.Sprintf("/proc/%d/net/tcp6", pid),
	} {
		raw, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(raw), "\n")[1:] {
			fields := strings.Fields(line)
			if len(fields) < 10 {
				continue
			}
			if fields[3] != "0A" {
				continue
			}
			inode, err := strconv.Atoi(fields[9])
			if err != nil || !inodes[inode] {
				continue
			}
			local := fields[1]
			i := strings.LastIndexByte(local, ':')
			if i < 0 {
				continue
			}
			port, err := strconv.ParseInt(local[i+1:], 16, 32)
			if err != nil {
				continue
			}
			p := int(port)
			if !seen[p] {
				seen[p] = true
				ports = append(ports, p)
			}
		}
	}
	return ports, nil
}

func readSocketInodes(pid int) (map[int]bool, error) {
	dir := fmt.Sprintf("/proc/%d/fd", pid)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	inodes := make(map[int]bool)
	for _, entry := range entries {
		link, err := os.Readlink(filepath.Join(dir, entry.Name()))
		if err != nil {
			continue
		}
		if !strings.HasPrefix(link, "socket:[") || !strings.HasSuffix(link, "]") {
			continue
		}
		raw := link[len("socket:[") : len(link)-1]
		inode, err := strconv.Atoi(raw)
		if err != nil {
			continue
		}
		inodes[inode] = true
	}
	return inodes, nil
}
