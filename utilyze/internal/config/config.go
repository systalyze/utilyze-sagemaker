package config

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type Config struct {
	ClientID string `json:"clientId"`
}

func configPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".systalyze", "utlz.json"), nil
}

func defaultConfig() Config {
	return Config{
		ClientID: randomID(),
	}
}

func sha256Hex(input string) string {
	hash := sha256.Sum256([]byte(input))
	return hex.EncodeToString(hash[:])
}

func randomID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func GenerateGpuID(gpuUUID string) string {
	return sha256Hex(gpuUUID)[:12]
}

func readMachineID() (string, error) {
	data, err := os.ReadFile("/etc/machine-id")
	if err != nil {
		return "", err
	}
	id := strings.TrimSpace(string(data))
	if id == "" {
		return "", os.ErrNotExist
	}
	return id, nil
}

func generateClientID(gpuUUIDs []string) string {
	if mid, err := readMachineID(); err == nil {
		return sha256Hex("mid:" + mid)[:24]
	}

	if len(gpuUUIDs) > 0 {
		sorted := append([]string(nil), gpuUUIDs...)
		sort.Strings(sorted)
		return sha256Hex("gpus:" + strings.Join(sorted, "|"))[:24]
	}

	return sha256Hex("rand:" + randomID())[:24]
}

func Load() Config {
	path, err := configPath()
	if err != nil {
		slog.Debug("could not use default config path; using default", "path", path, "err", err)
		return Config{ClientID: generateClientID(nil)}
	}

	cfgBytes, err := os.ReadFile(path)
	if err == nil {
		var c Config
		err := json.Unmarshal(cfgBytes, &c)
		if err == nil && c.ClientID != "" {
			return c
		}
		if err == nil {
			c.ClientID = randomID()
			if err := c.Save(); err != nil {
				slog.Debug("could not save config defaults; ignoring", "path", path, "err", err)
			}
			return c
		}
		slog.Debug("config parse failed; using default", "path", path, "err", err)
	} else {
		slog.Debug("config read failed; using default", "path", path, "err", err)
	}

	c := defaultConfig()
	if err := c.Save(); err != nil {
		slog.Debug("could not save config; using fallback identity", "path", path, "err", err)
		return Config{ClientID: generateClientID(nil)}
	}
	return c
}

func (c Config) Save() error {
	path, err := configPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	cfgBytes, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(cfgBytes, '\n'), 0600)
}
