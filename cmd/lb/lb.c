// SPDX-License-Identifier: MIT
// rinha-go v8 LB: epoll accept + SCM_RIGHTS to N backend UDS sockets.
// Replaces cmd/lb/main.go for the v8 submission. Built into /lb via Dockerfile.
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#ifndef SO_BUSY_POLL
#define SO_BUSY_POLL 46
#endif

#define MAX_BACKENDS 16
#define MAX_PREFIX   16384

static int env_int(const char *k, int dflt) {
    const char *v = getenv(k);
    if (!v || !*v) return dflt;
    return (int)strtol(v, NULL, 10);
}

static void die(const char *msg) {
    perror(msg);
    exit(1);
}

static int connect_uds(const char *path) {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    int sndbuf = 256 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    strncpy(sa.sun_path, path, sizeof(sa.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int wait_for_path(const char *path, int timeout_ms) {
    struct timespec ts = {0, 100 * 1000 * 1000};
    int waited = 0;
    while (waited < timeout_ms) {
        if (access(path, F_OK) == 0) return 0;
        nanosleep(&ts, NULL);
        waited += 100;
    }
    return -1;
}

// send_fd_with_bytes: send `clientfd` over `udsfd` with 2-byte LE length +
// prefix bytes as iov payload. Matches the worker recv decode.
static int send_fd_with_bytes(int udsfd, int clientfd,
                              const unsigned char *prefix, int plen) {
    if (plen > MAX_PREFIX) plen = MAX_PREFIX;
    unsigned char lenbuf[2];
    lenbuf[0] = (unsigned char)(plen & 0xff);
    lenbuf[1] = (unsigned char)((plen >> 8) & 0xff);

    struct iovec iov[2];
    iov[0].iov_base = lenbuf;
    iov[0].iov_len = 2;
    iov[1].iov_base = (void *)prefix;
    iov[1].iov_len = (size_t)plen;
    int iovlen = plen > 0 ? 2 : 1;

    union {
        struct cmsghdr cmsg;
        char buf[CMSG_SPACE(sizeof(int))];
    } u;
    memset(&u, 0, sizeof(u));

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = iov;
    msg.msg_iovlen = iovlen;
    msg.msg_control = u.buf;
    msg.msg_controllen = sizeof(u.buf);

    struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &clientfd, sizeof(int));

    for (;;) {
        ssize_t r = sendmsg(udsfd, &msg, MSG_NOSIGNAL);
        if (r >= 0) return 0;
        if (errno == EINTR) continue;
        return -1;
    }
}

// read_ready_prefix: drain any bytes immediately available from a non-blocking
// fd. Stops at EAGAIN or buffer full. Returns total bytes; -1 on fatal err.
static int read_ready_prefix(int fd, unsigned char *buf, int cap) {
    int n = 0;
    while (n < cap) {
        ssize_t r = read(fd, buf + n, (size_t)(cap - n));
        if (r > 0) {
            n += (int)r;
            continue;
        }
        if (r == 0) return n == 0 ? -1 : n;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        return -1;
    }
    return n;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    int port = env_int("LB_PORT", 9999);
    int backlog = env_int("LB_BACKLOG", 4096);
    int busy_poll = env_int("LB_BUSY_POLL_US", 50);

    const char *sockets = getenv("API_SOCKETS");
    if (!sockets || !*sockets) {
        fprintf(stderr, "API_SOCKETS empty\n");
        return 1;
    }

    char *list = strdup(sockets);
    if (!list) die("strdup");
    int backend_fds[MAX_BACKENDS];
    int nb = 0;
    char *save = NULL;
    for (char *tok = strtok_r(list, ",", &save); tok && nb < MAX_BACKENDS;
         tok = strtok_r(NULL, ",", &save)) {
        while (*tok == ' ') tok++;
        size_t L = strlen(tok);
        while (L > 0 && (tok[L - 1] == ' ' || tok[L - 1] == '\n')) {
            tok[--L] = 0;
        }
        if (L == 0) continue;
        if (wait_for_path(tok, 60000) != 0) {
            fprintf(stderr, "lb: wait %s timeout\n", tok);
            return 1;
        }
        int fd = connect_uds(tok);
        if (fd < 0) {
            fprintf(stderr, "lb: connect %s: %s\n", tok, strerror(errno));
            return 1;
        }
        fprintf(stderr, "lb: connected %s fd=%d\n", tok, fd);
        backend_fds[nb++] = fd;
    }
    free(list);
    if (nb == 0) {
        fprintf(stderr, "no backends\n");
        return 1;
    }

    int lfd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (lfd < 0) die("socket");
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(lfd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
    setsockopt(lfd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &one, sizeof(one));
    if (busy_poll > 0) {
        setsockopt(lfd, SOL_SOCKET, SO_BUSY_POLL, &busy_poll, sizeof(busy_poll));
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) die("bind");
    if (listen(lfd, backlog) < 0) die("listen");

    int epfd = epoll_create1(EPOLL_CLOEXEC);
    if (epfd < 0) die("epoll_create1");
    struct epoll_event ev = {.events = EPOLLIN, .data.fd = lfd};
    if (epoll_ctl(epfd, EPOLL_CTL_ADD, lfd, &ev) < 0) die("epoll_ctl");

    fprintf(stderr, "lb: listening :%d backlog=%d backends=%d busy_poll=%d\n",
            port, backlog, nb, busy_poll);

    int rr = 0;
    unsigned char prefix_buf[MAX_PREFIX];
    struct epoll_event evs[16];

    for (;;) {
        int n = epoll_wait(epfd, evs, 16, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            die("epoll_wait");
        }
        for (int e = 0; e < n; e++) {
            for (;;) {
                int cfd = accept4(lfd, NULL, NULL,
                                  SOCK_NONBLOCK | SOCK_CLOEXEC);
                if (cfd < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    if (errno == EINTR) continue;
                    break;
                }
                setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                setsockopt(cfd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));

                int plen = read_ready_prefix(cfd, prefix_buf, MAX_PREFIX);
                if (plen < 0) {
                    close(cfd);
                    continue;
                }
                int sent = 0;
                for (int k = 0; k < nb; k++) {
                    int i = (rr + k) % nb;
                    if (send_fd_with_bytes(backend_fds[i], cfd, prefix_buf,
                                           plen) == 0) {
                        rr = (i + 1) % nb;
                        sent = 1;
                        break;
                    }
                }
                close(cfd);
                (void)sent;
            }
        }
    }
}
