package service

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
)

func Stream(ctx context.Context, addr string, clientID string, handle func(Event) error) error {
	conn, err := dial(ctx, addr, clientID)
	if err != nil {
		return err
	}
	defer conn.CloseNow()

	for {
		var event Event
		err := wsjson.Read(ctx, conn, &event)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			switch websocket.CloseStatus(err) {
			case websocket.StatusNormalClosure, websocket.StatusGoingAway:
				return nil
			default:
				return fmt.Errorf("read live event: %w", err)
			}
		}
		if handle == nil {
			continue
		}
		if err := handle(event); err != nil {
			return err
		}
	}
}

func ServerAvailable(ctx context.Context, addr string, clientID string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 300*time.Millisecond)
	defer cancel()

	conn, err := dial(probeCtx, addr, clientID)
	if err != nil {
		return false
	}
	conn.CloseNow()
	return true
}

func dial(ctx context.Context, addr string, clientID string) (*websocket.Conn, error) {
	dialCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	conn, _, err := websocket.Dial(dialCtx, liveURLWithClientID(addr, clientID), nil)
	return conn, err
}

func liveURLWithClientID(addr string, clientID string) string {
	raw := LiveURL(addr)
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	q := u.Query()
	q.Set("client_id", clientID)
	u.RawQuery = q.Encode()
	return u.String()
}

func LiveURL(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		addr = DefaultAddr
	}

	switch {
	case strings.HasPrefix(addr, "http://"):
		addr = "ws://" + strings.TrimPrefix(addr, "http://")
	case strings.HasPrefix(addr, "https://"):
		addr = "wss://" + strings.TrimPrefix(addr, "https://")
	case !strings.HasPrefix(addr, "ws://") && !strings.HasPrefix(addr, "wss://"):
		addr = "ws://" + addr
	}

	u, err := url.Parse(addr)
	if err != nil {
		return addr
	}
	if u.Path == "" || u.Path == "/" {
		u.Path = LivePath
	}
	return u.String()
}
